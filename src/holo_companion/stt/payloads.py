from dataclasses import asdict
from pathlib import Path
from typing import Protocol, TypeAlias

from holo_companion.stt.base import SttConfig, STTProvider, Transcript
from holo_companion.stt.faster_whisper import FasterWhisperSttProvider
from holo_companion.stt.io import load_canonical_wav

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class SttProviderFactory(Protocol):
    def __call__(self, config: SttConfig) -> STTProvider: ...


def provider_from_config(config: SttConfig) -> STTProvider:
    return FasterWhisperSttProvider(config)


def transcript_payload(transcript: Transcript) -> JsonObject:
    return {
        "text": transcript.text,
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "audio_duration_seconds": transcript.audio_duration_seconds,
        "processing_seconds": transcript.processing_seconds,
        "real_time_factor": transcript.real_time_factor,
        "model_name": transcript.model_name,
        "device": transcript.device,
        "compute_type": transcript.compute_type,
        "segments": [asdict(segment) for segment in transcript.segments],
    }


def stt_wav_payload(paths: list[Path], config: SttConfig, language: str | None, factory: SttProviderFactory = provider_from_config) -> JsonObject:
    provider = factory(config)
    results = []
    for path in paths:
        transcript = provider.transcribe(load_canonical_wav(path), language_hint=language)
        results.append({"audio_path": str(path), "transcript": transcript_payload(transcript)})
    return {"results": results}
