import os
import random
import sys
from collections.abc import Sequence
from typing import TextIO

import anyio
import numpy as np
import soundfile

from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.tts.base import StyleHints, SynthesisRequest, TTSAudioChunk, TtsError, format_empty_audio_diagnostic

CURATED_GREETINGS: tuple[str, ...] = (
    "[shyly] H-halo, Master... [softly] akhirnya kamu datang juga. Hehe... aku udah nungguin.",
    "[shyly] Eh, Master... [softly] udah balik ya? Aku... kangen pengen ngobrol, hehe.",
    "[shyly] H-hai, Master... [softly] Masuk aja, aku udah siap kok buat nemenin kamu.",
    "[shyly] Master... [softly] akhirnya kamu luang juga. Jangan capek-capek ya, yuk ngobrol sebentar.",
)

DEFAULT_GREETING_STYLE = StyleHints(emotion="shy", intensity=0.5, speaking_style="soft")


def select_startup_greeting(
    choices: Sequence[str] = CURATED_GREETINGS,
    random_fn=random.choice,
) -> SynthesisRequest:
    text = random_fn(choices)
    return SynthesisRequest(text=text, style=DEFAULT_GREETING_STYLE)


async def play_startup_greeting(tts_provider, playback, stdout: TextIO) -> None:
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
    except TtsError as error:
        diag = error.diagnostic
        if diag is not None and diag.error == "empty_audio":
            print(
                format_empty_audio_diagnostic(
                    turn_id=0,
                    segment_index=0,
                    text_length=len(request.text),
                    retry_attempt=0,
                    diagnostic=diag,
                    retry_triggered=True,
                    cancellation_or_staleness_active=False,
                ),
                file=sys.stderr,
                flush=True,
            )
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
            except TtsError as retry_err:
                print(
                    format_empty_audio_diagnostic(
                        turn_id=0,
                        segment_index=0,
                        text_length=len(request.text),
                        retry_attempt=1,
                        diagnostic=retry_err.diagnostic,
                        retry_triggered=False,
                        cancellation_or_staleness_active=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                stop_reason = str(retry_err)
                stdout.write(f"greeting error: {retry_err}\n")
            except Exception as retry_other_err:
                stop_reason = str(retry_other_err)
                stdout.write(f"greeting error: {retry_other_err}\n")
        elif diag is not None and diag.error == "quota_exceeded":
            stop_reason = "ElevenLabs quota exhausted"
            stdout.write("greeting error: ElevenLabs quota exhausted\n")
        else:
            stop_reason = str(error)
            stdout.write(f"greeting error: {error}\n")
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
