import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

from holo_companion.stt.base import SttConfig, SttError
from holo_companion.stt.io import load_canonical_wav
from holo_companion.stt.payloads import JsonObject, SttProviderFactory, provider_from_config, transcript_payload

JsonManifestValue: TypeAlias = str | list["ManifestEntry"]


class ManifestEntry(TypedDict):
    audio_path: str
    expected_text: str
    language: str


@dataclass(frozen=True, slots=True)
class Score:
    word_error_rate: float
    char_error_rate: float
    exact: bool


def benchmark_payload(manifest_path: Path, config: SttConfig, factory: SttProviderFactory = provider_from_config) -> JsonObject:
    entries = _load_entries(manifest_path)
    provider = factory(config)
    results = []
    total_audio = 0.0
    total_processing = 0.0
    exact_count = 0
    wer_total = 0.0
    cer_total = 0.0
    for entry in entries:
        audio_path = (manifest_path.parent / entry["audio_path"]).resolve()
        transcript = provider.transcribe(load_canonical_wav(audio_path), language_hint=entry["language"])
        score = score_transcript(transcript.text, entry["expected_text"])
        total_audio += transcript.audio_duration_seconds
        total_processing += transcript.processing_seconds
        exact_count += int(score.exact)
        wer_total += score.word_error_rate
        cer_total += score.char_error_rate
        results.append({"audio_path": str(audio_path), "expected_text": entry["expected_text"], "language": entry["language"], "transcript": transcript_payload(transcript), "score": asdict(score)})
    count = len(entries)
    return {
        "manifest_path": str(manifest_path),
        "results": results,
        "aggregate": {
            "sample_count": count,
            "word_error_rate": wer_total / count,
            "char_error_rate": cer_total / count,
            "exact_rate": exact_count / count,
            "total_audio_duration_seconds": total_audio,
            "total_processing_seconds": total_processing,
            "real_time_factor": total_processing / total_audio,
        },
    }


def score_transcript(raw_text: str, expected_text: str) -> Score:
    actual_words = _normalize(raw_text).split()
    expected_words = _normalize(expected_text).split()
    actual_chars = list(_normalize(raw_text))
    expected_chars = list(_normalize(expected_text))
    return Score(
        word_error_rate=_rate(_edit_distance(actual_words, expected_words), len(expected_words)),
        char_error_rate=_rate(_edit_distance(actual_chars, expected_chars), len(expected_chars)),
        exact=actual_words == expected_words,
    )


def _load_entries(path: Path) -> list[ManifestEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SttError("benchmark manifest must be a JSON object")
    entries = raw["entries"]
    if not isinstance(entries, list):
        raise SttError("benchmark manifest entries must be a list")
    if not entries:
        raise SttError("benchmark manifest must contain entries")
    return entries


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _rate(distance: int, expected_length: int) -> float:
    if expected_length == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / expected_length


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            substitution = 0 if left_value == right_value else 1
            current.append(min(previous[right_index] + 1, current[right_index - 1] + 1, previous[right_index - 1] + substitution))
        previous = current
    return previous[-1]
