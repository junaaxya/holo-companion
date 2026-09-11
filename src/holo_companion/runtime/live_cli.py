import math
import time
from pathlib import Path
from typing import TextIO

import anyio

from holo_companion.audio.capture import AudioBackend, resolve_input_device
from holo_companion.audio.capture import SoundFileWaveWriter
from holo_companion.audio.raw_capture import RawCaptureDumper
from holo_companion.cli_commands import parse_input_device
from holo_companion.audio.playback import SoundDevicePlaybackSink
from holo_companion.audio.types import CAPTURE_CHANNELS, CAPTURE_DTYPE, CAPTURE_FRAME_SAMPLES, CAPTURE_SAMPLE_RATE, AudioCliError
from holo_companion.llm.config import load_llm_config
from holo_companion.llm.openai_compatible import provider_from_config as llm_provider_from_config
from holo_companion.runtime.barge_in import AdaptiveBargeInGate
from holo_companion.runtime.greeting import select_startup_greeting
from holo_companion.runtime.live_pump import CaptureVadPump, InterruptionGuard
from holo_companion.runtime.orchestrator import RuntimeOrchestrator
from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.runtime.turn import TurnResult
from holo_companion.stt.collector import UtteranceCollector
from holo_companion.stt.payloads import provider_from_config as stt_provider_from_config
from holo_companion.tts.base import TTSAudioChunk
from holo_companion.tts.config import load_elevenlabs_tts_config
from holo_companion.tts.elevenlabs import provider_from_config as tts_provider_from_config
from holo_companion.vad.base import TurnDetector, VadPolicy
from holo_companion.vad.silero import create_silero_vad_provider


async def run_talk(args, backend: AudioBackend, stdout: TextIO) -> None:
    requested_device = parse_input_device(args.input_device)
    selection = resolve_input_device(requested_device, backend)
    backend.check_input_settings(selection.resolved.index, int(CAPTURE_SAMPLE_RATE), CAPTURE_CHANNELS, CAPTURE_DTYPE)
    policy = VadPolicy(args.min_speech_ms, args.min_silence_ms, args.speech_pad_ms, args.threshold)
    stdout.write(f"requested input device: {selection.requested}\n")
    stdout.write(f"resolved input device: {selection.resolved.name} ({int(selection.resolved.index)})\n")
    stdout.write("capture samplerate: 16000\ncapture channels: 1\ncapture dtype: float32\nframe samples: 512\n")
    guard = InterruptionGuard(args.playback_guard_ms, time.monotonic)
    barge_in_gate = AdaptiveBargeInGate()
    playback = SoundDevicePlaybackSink(
        on_first_consumed=lambda _admission: guard.mark_playback_consumed(),
        on_samples_consumed=barge_in_gate.observe_playback,
    )
    stt = stt_provider_from_config(args.stt_config)
    stt_load_seconds = await anyio.to_thread.run_sync(stt.preload)
    _write_latency(stdout, "STT model load", stt_load_seconds)
    tts_provider = tts_provider_from_config(load_elevenlabs_tts_config(), time.monotonic)
    runtime = RuntimeOrchestrator(
        stt=stt,
        llm=llm_provider_from_config(load_llm_config(), time.monotonic),
        tts=tts_provider,
        playback=playback,
    )
    await _play_startup_greeting(tts_provider, playback, stdout)
    async def report_turn(result: TurnResult) -> None:
        if result.transcript is not None:
            stdout.write(f"transcript: {result.transcript}\n")
        if result.utterance_duration_seconds is not None:
            _write_latency(stdout, "VAD utterance", result.utterance_duration_seconds)
        if result.utterance_sample_count is not None:
            stdout.write(f"turn_id: {result.turn_id}\n")
            stdout.write(f"utterance sample count: {result.utterance_sample_count}\n")
            stdout.write(f"utterance peak: {result.utterance_peak:.6f}\n")
            stdout.write(f"utterance RMS: {result.utterance_rms:.6f}\n")
            stdout.write(f"VAD frames: {result.vad_start_frame_index} -> {result.vad_end_frame_index}\n")
        metrics = result.metrics
        _write_latency(stdout, "speech_end -> STT complete", metrics.speech_end_to_transcript_seconds)
        _write_latency(stdout, "speech_end -> first LLM token", metrics.speech_end_to_first_token_seconds)
        _write_latency(stdout, "speech_end -> first TTS audio", metrics.speech_end_to_first_audio_seconds)
        _write_latency(stdout, "speech_end -> first playback acknowledgement", metrics.speech_end_to_first_playback_seconds)
        _write_latency(stdout, "STT wait", metrics.stt_wait_seconds)
        _write_latency(stdout, "STT invocation", metrics.stt_invocation_seconds)
        _write_latency(stdout, "STT total", metrics.speech_end_to_transcript_seconds)
        _write_latency(stdout, "LLM invocation -> provider request", metrics.llm_invocation_to_provider_request_seconds)
        _write_latency(stdout, "LLM provider request -> first raw delta", metrics.llm_provider_request_to_first_raw_delta_seconds)
        _write_latency(stdout, "LLM provider request -> first usable delta", metrics.llm_provider_request_to_first_usable_delta_seconds)
        if result.stt_inference_seconds is not None:
            _write_latency(stdout, "STT inference", result.stt_inference_seconds)
        if result.error is not None:
            stdout.write(f"turn error: {result.error.kind} ({result.error.stage})\n")
            diagnostic = result.error.diagnostic
            if diagnostic is not None:
                stdout.write(
                    f"provider: {diagnostic.provider}; exception: {diagnostic.exception_class}; "
                    f"message: {diagnostic.message}; http_status: {diagnostic.http_status}; "
                    f"provider_code: {diagnostic.provider_code}; recoverable: {diagnostic.recoverable}\n"
                )
            interruption = result.error.interruption
            if interruption is not None:
                stdout.write(
                    f"interruption: previous_turn_id={interruption.previous_turn_id} "
                    f"previous_state={interruption.previous_state} "
                    f"new_speech_frame={interruption.new_speech_frame} "
                    f"reason={interruption.reason}\n"
                )
        if result.text_diagnostic is not None:
            text = result.text_diagnostic
            stdout.write(f"LLM assistant text: {text.llm_text}\n")
            for segment in text.emitted_segments:
                stdout.write(f"segment emitted [{segment.index}]: {segment.text}\n")
            for segment in text.tts_segments:
                stdout.write(f"TTS submitted [{segment.index}]: {segment.text}\n")
            stdout.write(f"segmenter final buffer flushed: {text.final_buffer_flushed}\n")
            stdout.write(f"LLM finish reason: {text.llm_finish_reason}\n")

    dump = _utterance_dumper(args.dump_utterances_dir, stdout)
    raw_dump = None if args.dump_raw_capture is None else RawCaptureDumper(args.dump_raw_capture)
    if raw_dump is not None:
        stdout.write(f"raw capture dump: {raw_dump.path}\n")
    pump = CaptureVadPump(
        backend, create_silero_vad_provider(policy), TurnDetector(policy),
        UtteranceCollector(history_frames=18), runtime, selection.resolved.index, guard,
        on_turn_complete=report_turn,
        dump_utterance=dump,
        raw_capture_dumper=raw_dump,
        barge_in_gate=barge_in_gate,
    )
    stdout.write("STATE: LISTENING\n")
    try:
        await pump.run()
    finally:
        await runtime.aclose()
        stdout.write(f"capture overruns: {pump.diagnostics.capture_overruns}\n")
        stdout.write(f"playback starvations: {playback.diagnostics.playback_starvations}\n")
        stdout.write(f"device output underflows: {playback.diagnostics.device_output_underflows}\n")
        stdout.write(f"playback backpressure waits: {playback.diagnostics.backpressure_waits}\n")
        stdout.write(f"self-VAD suppressions: {pump.diagnostics.self_trigger_suppressions}\n")
        stdout.write(f"barge-in accepted: {pump.diagnostics.barge_in_accepted}\n")
        stdout.write(f"barge-in rejected noise: {pump.diagnostics.barge_in_rejected_noise}\n")
        stdout.write(f"barge-in rejected short: {pump.diagnostics.barge_in_rejected_short}\n")
        _write_metric(stdout, "barge-in echo correlation", pump.diagnostics.barge_in_echo_correlation)
        _write_metric(stdout, "barge-in echo best delay ms", barge_in_gate.last_echo_best_delay_ms)
        _write_metric(stdout, "barge-in playback reference RMS", barge_in_gate.last_playback_reference_rms)
        stdout.write(f"barge-in rejected echo: {pump.diagnostics.barge_in_rejected_echo}\n")
        _write_metric(stdout, "barge-in candidate RMS", barge_in_gate.last_candidate_rms)
        _write_metric(stdout, "barge-in ambient noise RMS", barge_in_gate.last_ambient_noise_rms)
        _write_metric(stdout, "barge-in RMS ratio", barge_in_gate.last_rms_ratio)
        _write_metric(stdout, "barge-in SNR dB", None if barge_in_gate.last_rms_ratio is None else 20.0 * math.log10(barge_in_gate.last_rms_ratio))
        _write_metric(stdout, "barge-in VAD confidence", barge_in_gate.last_vad_confidence)
        stdout.write(f"barge-in final reject reason: {barge_in_gate.last_reject_reason or 'accepted'}\n")
        stdout.write(f"interruptions: {sum(result.error is not None and result.error.kind.value == 'cancelled' for result in runtime.turn_results)}\n")


def _write_latency(stdout: TextIO, label: str, seconds: float | None) -> None:
    if seconds is not None:
        stdout.write(f"{label}: {seconds:.3f} s\n")


def _write_metric(stdout: TextIO, label: str, value: float | None) -> None:
    if value is not None:
        stdout.write(f"{label}: {value:.3f}\n")


def _utterance_dumper(directory: Path | None, stdout: TextIO):
    if directory is None:
        return None
    writer = SoundFileWaveWriter()
    sequence = 0

    def dump(utterance) -> None:
        nonlocal sequence
        sequence += 1
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"turn-{sequence:04d}.wav"
        writer.write_pcm16(path, utterance.samples, utterance.sample_rate_hz)
        stdout.write(f"utterance dump: {path}\n")

    return dump


async def _play_startup_greeting(tts_provider, playback, stdout: TextIO) -> None:
    import os
    import numpy as np
    import soundfile
    request = select_startup_greeting()
    mapped_text = getattr(tts_provider, "emotion_mapper", None)
    after_mapping = mapped_text.map_text(request.text, request.style).text if mapped_text else request.text
    stdout.write(f"greeting: {request.text}\n")
    stdout.write(f"startup original text: {request.text}\n")
    stdout.write(f"startup text after EmotionMapper: {after_mapping}\n")
    stdout.write(f"exact text submitted to ElevenLabs: {after_mapping}\n")
    stdout.write(f"startup TTS text: {request.text}\n")
    debug_audio = os.environ.get("HOLO_DEBUG_TTS_AUDIO") == "1"
    provider_audio_chunks: list[np.ndarray] = []
    playback_audio_chunks: list[np.ndarray] = []
    sample_rate_hz = 24000

    def capture_playback_samples(samples: np.ndarray, rate: int) -> None:
        nonlocal sample_rate_hz
        sample_rate_hz = rate
        if debug_audio:
            playback_audio_chunks.append(samples.copy())

    orig_consumed_cb = getattr(playback, "on_samples_consumed", None)

    def combined_consumed_cb(samples: np.ndarray, rate: int) -> None:
        capture_playback_samples(samples, rate)
        if orig_consumed_cb is not None:
            orig_consumed_cb(samples, rate)

    if hasattr(playback, "on_samples_consumed"):
        playback.on_samples_consumed = combined_consumed_cb

    chunk_count = 0
    received_samples = 0
    queued_samples = 0
    stream_completed = False
    stop_reason: str | None = None
    try:
        sequence = 0
        async for event in tts_provider.stream(request):
            if isinstance(event, TTSAudioChunk):
                chunk_count += 1
                received_samples += event.samples.size
                if debug_audio:
                    provider_audio_chunks.append(event.samples.copy())
                sample_rate_hz = event.audio_format.sample_rate_hz
                await playback.write_and_start(PlaybackAdmission(0, 0, event))
                queued_samples += event.samples.size
                sequence += 1
        stream_completed = True
        if hasattr(playback, "wait_until_drained"):
            await playback.wait_until_drained()
    except anyio.get_cancelled_exc_class():
        stop_reason = "cancelled"
        raise
    except Exception as error:
        stop_reason = str(error)
        stdout.write(f"greeting error: {error}\n")
    finally:
        if hasattr(playback, "on_samples_consumed"):
            playback.on_samples_consumed = orig_consumed_cb
        consumed_samples = getattr(playback, "total_consumed_samples", sum(chunk.size for chunk in playback_audio_chunks))
        stdout.write(f"provider audio chunk count: {chunk_count}\n")
        stdout.write(f"provider total samples: {received_samples}\n")
        stdout.write(f"provider total duration: {received_samples / sample_rate_hz:.3f} s\n")
        stdout.write(f"provider stream completed: {stream_completed}\n")
        stdout.write(f"playback queued samples: {queued_samples}\n")
        stdout.write(f"playback consumed samples: {consumed_samples}\n")
        stdout.write(f"playback drain completed: {stream_completed and stop_reason is None}\n")
        stdout.write(f"stop/cancel/interruption reason: {stop_reason or 'none'}\n")

        if debug_audio:
            try:
                if provider_audio_chunks:
                    all_provider_pcm = np.concatenate(provider_audio_chunks)
                    soundfile.write("/tmp/holo-provider.wav", all_provider_pcm, sample_rate_hz, format="WAV", subtype="PCM_16")
                    stdout.write(f"debug saved: /tmp/holo-provider.wav ({all_provider_pcm.size} samples)\n")
                if playback_audio_chunks:
                    all_playback_pcm = np.concatenate(playback_audio_chunks)
                    soundfile.write("/tmp/holo-playback.wav", all_playback_pcm, sample_rate_hz, format="WAV", subtype="PCM_16")
                    stdout.write(f"debug saved: /tmp/holo-playback.wav ({all_playback_pcm.size} samples)\n")
            except Exception as dump_err:
                stdout.write(f"debug dump error: {dump_err}\n")


def run_talk_cli(args, backend: AudioBackend, stdout: TextIO, stderr: TextIO) -> int:
    try:
        anyio.run(run_talk, args, backend, stdout)
    except KeyboardInterrupt:
        return 0
    except (AudioCliError, OSError) as error:
        print(f"error: {error}", file=stderr)
        return 2
    return 0
