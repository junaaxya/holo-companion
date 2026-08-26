import contextlib
import math
from time import perf_counter_ns
from argparse import Namespace
from dataclasses import asdict
from typing import Callable, assert_never

from holo_companion.audio.capture import AudioBackend, resolve_input_device
from holo_companion.audio.types import CAPTURE_CHANNELS, CAPTURE_DTYPE, CAPTURE_FRAME_MS, CAPTURE_FRAME_SAMPLES, CAPTURE_SAMPLE_RATE, AudioCliError
from holo_companion.cli_support import parse_stt_config
from holo_companion.stt.collector import UtteranceCollector
from holo_companion.stt.payloads import JsonObject, SttProviderFactory, provider_from_config, transcript_payload
from holo_companion.vad.base import PublicSpeechEvent, SpeechEnded, SpeechStarted, TurnDetector, VadPolicy, VadProvider

VadProviderFactory = Callable[[VadPolicy], VadProvider]


def vad_test_payload(
    args: Namespace,
    backend: AudioBackend,
    vad_provider_factory: VadProviderFactory,
    stt_provider_factory: SttProviderFactory = provider_from_config,
) -> JsonObject:
    input_device = _parse_input_device(args.input_device)
    policy = VadPolicy(
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        threshold=args.threshold,
    )
    selection = resolve_input_device(input_device, backend)
    backend.check_input_settings(selection.resolved.index, samplerate=int(CAPTURE_SAMPLE_RATE), channels=CAPTURE_CHANNELS, dtype=CAPTURE_DTYPE)
    requested_frames = math.ceil(float(args.seconds) * int(CAPTURE_SAMPLE_RATE) / CAPTURE_FRAME_SAMPLES)
    provider = vad_provider_factory(policy)
    stt_provider = stt_provider_factory(parse_stt_config(args)) if bool(getattr(args, "transcribe", False)) else None
    collector = UtteranceCollector(history_frames=math.ceil(policy.min_speech_ms / CAPTURE_FRAME_MS) + 2) if stt_provider is not None else None
    detector = TurnDetector(policy)
    events: list[dict[str, str | int | float]] = []
    transcripts: list[JsonObject] = []
    process_times_ms: list[float] = []
    try:
        stream = backend.frame_stream(int(CAPTURE_SAMPLE_RATE), CAPTURE_CHANNELS, CAPTURE_DTYPE, selection.resolved.index, CAPTURE_FRAME_SAMPLES)
        with contextlib.closing(stream):
            for _ in range(requested_frames):
                frame = next(stream)
                process_started_ns = perf_counter_ns()
                speech_probability = provider.process(frame)
                process_times_ms.append((perf_counter_ns() - process_started_ns) / 1_000_000)
                frame_events = list(detector.process(frame, speech_probability))
                events.extend(speech_event_payload(event) for event in frame_events)
                if collector is not None and stt_provider is not None:
                    utterance = collector.process(frame, frame_events)
                    if utterance is not None:
                        transcripts.append(transcript_payload(stt_provider.transcribe(utterance, language_hint=args.language)))
    except StopIteration as error:
        raise AudioCliError("audio frame stream ended before requested duration") from error
    finally:
        provider.reset()
    diagnostics = detector.diagnostics
    payload = {
        "requested_input_device": selection.requested,
        "resolved_input_device": {
            "index": int(selection.resolved.index),
            "name": selection.resolved.name,
            "max_input_channels": selection.resolved.max_input_channels,
            "max_output_channels": selection.resolved.max_output_channels,
            "default_samplerate": selection.resolved.default_samplerate,
        },
        "format": {"samplerate_hz": int(CAPTURE_SAMPLE_RATE), "channels": CAPTURE_CHANNELS, "dtype": CAPTURE_DTYPE, "frame_samples": CAPTURE_FRAME_SAMPLES, "frame_ms": CAPTURE_FRAME_MS},
        "policy": asdict(policy),
        "events": events,
        "diagnostics": {
            "requested_seconds": float(args.seconds),
            "requested_frames": requested_frames,
            "processed_frames": diagnostics.processed_frames,
            "processed_samples": diagnostics.processed_frames * CAPTURE_FRAME_SAMPLES,
            "processed_duration_seconds": diagnostics.processed_frames * CAPTURE_FRAME_SAMPLES / int(CAPTURE_SAMPLE_RATE),
            "speech_started_count": diagnostics.speech_started_count,
            "speech_ended_count": diagnostics.speech_ended_count,
            "rejected_speech_candidates": diagnostics.rejected_speech_candidates,
            "last_end_latency_ms": diagnostics.last_end_latency_ms,
            "mean_provider_process_ms": sum(process_times_ms) / len(process_times_ms) if process_times_ms else 0.0,
            "max_provider_process_ms": max(process_times_ms, default=0.0),
            "queue_overrun": False,
        },
    }
    if stt_provider is not None:
        payload["transcripts"] = transcripts
    return payload


def speech_event_payload(event: PublicSpeechEvent) -> dict[str, str | int | float]:
    match event:
        case SpeechStarted() as event:
            return {
                "type": event.type,
                "start_frame_index": event.start_frame_index,
                "detected_frame_index": event.detected_frame_index,
                "start_time_ms": event.start_time_ms,
            }
        case SpeechEnded() as event:
            return {
                "type": event.type,
                "start_frame_index": event.start_frame_index,
                "end_frame_index": event.end_frame_index,
                "detected_frame_index": event.detected_frame_index,
                "duration_ms": event.duration_ms,
                "end_latency_ms": event.end_latency_ms,
            }
        case unreachable:
            assert_never(unreachable)


def _parse_input_device(input_device: str) -> str:
    normalized = input_device.strip()
    if normalized == "":
        raise AudioCliError("--input-device must be 'auto' or a device-name query")
    if normalized.isdecimal():
        raise AudioCliError("--input-device accepts a device-name query, not a numeric device id")
    return normalized
