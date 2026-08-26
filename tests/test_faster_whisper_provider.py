from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest

from holo_companion.stt.base import SttConfig, SttError, Utterance
from holo_companion.stt.faster_whisper import FasterWhisperSttProvider


@dataclass(frozen=True, slots=True)
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class FakeInfo:
    language: str
    language_probability: float


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeModel:
    consumed_at: list[float]
    languages: list[str | None]

    def transcribe(self, samples: np.ndarray, language: str | None, task: str, vad_filter: bool) -> tuple[Iterator[FakeSegment], FakeInfo]:
        assert samples.dtype == np.float32
        self.languages.append(language)
        assert task == "transcribe"
        assert vad_filter is False

        def segments() -> Iterator[FakeSegment]:
            self.consumed_at.append(20.0)
            yield FakeSegment(0.0, 0.4, " Halo")
            yield FakeSegment(0.4, 0.8, " dunia")

        return segments(), FakeInfo(language="id", language_probability=0.95)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeLoader:
    calls: list[tuple[str, str, str, str, bool, int, int]]
    model: FakeModel

    def __call__(self, model_name: str, device: str, compute_type: str, download_root: str, local_files_only: bool, cpu_threads: int, num_workers: int) -> FakeModel:
        self.calls.append((model_name, device, compute_type, download_root, local_files_only, cpu_threads, num_workers))
        return self.model


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeClock:
    values: list[float]

    def __call__(self) -> float:
        return self.values.pop(0)


def test_faster_whisper_constructs_exact_model_config_and_maps_segments(tmp_path) -> None:
    # Given
    model = FakeModel(consumed_at=[], languages=[])
    loader = FakeLoader(calls=[], model=model)
    provider = FasterWhisperSttProvider(
        SttConfig(cache_dir=tmp_path, local_files_only=True, cpu_threads=2, num_workers=3),
        loader=loader,
        monotonic_seconds=FakeClock([10.0, 21.0]),
    )
    utterance = Utterance.from_samples(np.zeros(16_000, dtype=np.float32))

    # When
    transcript = provider.transcribe(utterance, language_hint="id")

    # Then
    assert loader.calls == [("large-v3-turbo", "cpu", "int8", str(tmp_path), True, 2, 3)]
    assert model.consumed_at == [20.0]
    assert model.languages == ["id"]
    assert transcript.text == " Halo dunia"
    assert transcript.language == "id"
    assert transcript.language_probability == 0.95
    assert transcript.audio_duration_seconds == 1.0
    assert transcript.processing_seconds == 11.0
    assert transcript.real_time_factor == 11.0
    assert transcript.model_name == "large-v3-turbo"
    assert transcript.device == "cpu"
    assert transcript.compute_type == "int8"
    assert [(segment.start_seconds, segment.end_seconds, segment.text) for segment in transcript.segments] == [(0.0, 0.4, " Halo"), (0.4, 0.8, " dunia")]


def test_faster_whisper_translates_loader_value_error(tmp_path) -> None:
    # Given
    def failing_loader(model_name: str, device: str, compute_type: str, download_root: str, local_files_only: bool, cpu_threads: int, num_workers: int) -> FakeModel:
        raise ValueError("missing model")

    provider = FasterWhisperSttProvider(SttConfig(cache_dir=tmp_path), loader=failing_loader)
    utterance = Utterance.from_samples(np.zeros(512, dtype=np.float32))

    # When / Then
    with pytest.raises(SttError, match="failed to load Faster-Whisper model"):
        provider.transcribe(utterance)


def test_faster_whisper_translates_transcribe_runtime_error(tmp_path) -> None:
    # Given
    @dataclass(frozen=True, slots=True)
    class RuntimeFailModel:
        def transcribe(self, samples: np.ndarray, language: str | None, task: str, vad_filter: bool) -> tuple[Iterator[FakeSegment], FakeInfo]:
            raise RuntimeError("runtime failed")

    def loader(model_name: str, device: str, compute_type: str, download_root: str, local_files_only: bool, cpu_threads: int, num_workers: int) -> RuntimeFailModel:
        return RuntimeFailModel()

    provider = FasterWhisperSttProvider(SttConfig(cache_dir=tmp_path), loader=loader)
    utterance = Utterance.from_samples(np.zeros(512, dtype=np.float32))

    # When / Then
    with pytest.raises(SttError, match="Faster-Whisper transcription failed"):
        provider.transcribe(utterance)


def test_faster_whisper_maps_auto_language_hint_to_model_detection(tmp_path) -> None:
    # Given
    model = FakeModel(consumed_at=[], languages=[])
    provider = FasterWhisperSttProvider(
        SttConfig(cache_dir=tmp_path),
        loader=FakeLoader(calls=[], model=model),
        monotonic_seconds=FakeClock([1.0, 2.0]),
    )
    utterance = Utterance.from_samples(np.zeros(16_000, dtype=np.float32))

    # When
    provider.transcribe(utterance, language_hint="auto")

    # Then
    assert model.consumed_at == [20.0]
    assert model.languages == [None]
