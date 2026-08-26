from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from holo_companion.audio.types import CAPTURE_SAMPLE_RATE

SttModelName: TypeAlias = Literal["large-v3-turbo", "large-v3", "small"]
SttDevice: TypeAlias = Literal["cpu"]
SttComputeType: TypeAlias = Literal["int8"]
ALLOWED_MODELS: Final = frozenset(("large-v3-turbo", "large-v3", "small"))
DEFAULT_CACHE_DIR: Final = Path("~/.cache/holo-companion/models")


@dataclass(frozen=True, slots=True)
class SttError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class SttConfig:
    model_name: str = "large-v3-turbo"
    cache_dir: Path = DEFAULT_CACHE_DIR
    local_files_only: bool = False
    cpu_threads: int = 0
    num_workers: int = 1
    device: SttDevice = "cpu"
    compute_type: SttComputeType = "int8"

    def __post_init__(self) -> None:
        if self.model_name not in ALLOWED_MODELS:
            raise SttError(f"unsupported STT model: {self.model_name}")
        if self.cpu_threads < 0:
            raise SttError("STT cpu_threads must be at least 0")
        if self.num_workers < 1:
            raise SttError("STT num_workers must be at least 1")
        object.__setattr__(self, "cache_dir", self.cache_dir.expanduser())


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str
    language_probability: float
    audio_duration_seconds: float
    processing_seconds: float
    real_time_factor: float
    model_name: str
    device: SttDevice
    compute_type: SttComputeType
    segments: tuple[TranscriptSegment, ...]


@dataclass(frozen=True, slots=True)
class Utterance:
    samples: NDArray[np.float32]
    sample_rate_hz: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None

    @classmethod
    def from_samples(
        cls,
        samples: NDArray[np.float32],
        *,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
    ) -> "Utterance":
        if samples.ndim != 1:
            raise SttError("STT utterance audio must be rank-one mono float32")
        if samples.dtype != np.float32:
            raise SttError("STT utterance audio must be float32")
        if not samples.flags.c_contiguous:
            raise SttError("STT utterance audio must be contiguous")
        if samples.size == 0:
            raise SttError("STT utterance audio must be nonempty")
        if not np.all(np.isfinite(samples)):
            raise SttError("STT utterance audio must be finite")
        return cls(samples=samples, sample_rate_hz=int(CAPTURE_SAMPLE_RATE), start_frame_index=start_frame_index, end_frame_index=end_frame_index)

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate_hz


class STTProvider(Protocol):
    def transcribe(self, utterance: Utterance, language_hint: str | None = None) -> Transcript: ...
