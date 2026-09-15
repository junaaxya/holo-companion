from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from holo_companion.llm.base import CancellationHandle

TtsPcmDtype: TypeAlias = Literal["float32"]


@unique
class TtsErrorKind(StrEnum):
    CONFIG = "config"
    CANCELLED = "cancelled"
    PROVIDER = "provider"


@unique
class SynthesisStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TtsProviderDiagnostic:
    stage: str
    status_code: int | None = None
    close_code: int | None = None
    error: str | None = None
    error_class: str | None = None
    message: str | None = None
    model: str | None = None
    content_type: str | None = None
    received_byte_count: int = 0
    decoded_sample_count: int = 0
    stream_completed_normally: bool = False


def format_empty_audio_diagnostic(
    turn_id: int,
    segment_index: int,
    text_length: int,
    retry_attempt: int,
    diagnostic: TtsProviderDiagnostic | None,
    retry_triggered: bool,
    cancellation_or_staleness_active: bool,
) -> str:
    status_code = diagnostic.status_code if diagnostic else None
    content_type = diagnostic.content_type if diagnostic else None
    received_bytes = diagnostic.received_byte_count if diagnostic else 0
    decoded_samples = diagnostic.decoded_sample_count if diagnostic else 0
    stream_completed = diagnostic.stream_completed_normally if diagnostic else False
    return (
        "empty_audio diagnostic:\n"
        f"  turn_id: {turn_id}\n"
        f"  TTS segment index: {segment_index}\n"
        f"  segment text length: {text_length}\n"
        f"  retry attempt: {retry_attempt}\n"
        f"  HTTP status: {status_code}\n"
        f"  response content-type: {content_type}\n"
        f"  total raw response bytes received: {received_bytes}\n"
        f"  decoded PCM sample count: {decoded_samples}\n"
        f"  whether HTTP stream completed normally: {stream_completed}\n"
        f"  whether retry was triggered: {retry_triggered}\n"
        f"  whether cancellation/staleness was active: {cancellation_or_staleness_active}"
    )


class TtsError(Exception):
    __slots__ = ("_frozen", "diagnostic", "kind", "message")

    def __init__(self, kind: TtsErrorKind, message: str, diagnostic: TtsProviderDiagnostic | None = None) -> None:
        super().__init__(kind, message, diagnostic)
        super().__setattr__("kind", kind)
        super().__setattr__("message", message)
        super().__setattr__("diagnostic", diagnostic)
        super().__setattr__("_frozen", True)

    def __setattr__(self, name: str, value: TtsErrorKind | TtsProviderDiagnostic | str | bool | None) -> None:
        if name in {"__traceback__", "__cause__", "__context__", "__suppress_context__"}:
            super().__setattr__(name, value)
            return
        if getattr(self, "_frozen", False):
            raise AttributeError("TtsError is immutable")
        super().__setattr__(name, value)

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.message}"


@dataclass(frozen=True, slots=True)
class StyleHints:
    emotion: str | None = None
    energy: float | None = None
    intensity: float | None = None
    speaking_style: str | None = None

    def __post_init__(self) -> None:
        if self.emotion is not None and self.emotion.strip() == "":
            raise TtsError(TtsErrorKind.CONFIG, "style emotion must be nonblank when set")
        if self.speaking_style is not None and self.speaking_style.strip() == "":
            raise TtsError(TtsErrorKind.CONFIG, "style speaking_style must be nonblank when set")
        if self.energy is not None and (not np.isfinite(self.energy) or not 0.0 <= self.energy <= 1.0):
            raise TtsError(TtsErrorKind.CONFIG, "style energy must be between 0.0 and 1.0")
        if self.intensity is not None and (not np.isfinite(self.intensity) or not 0.0 <= self.intensity <= 1.0):
            raise TtsError(TtsErrorKind.CONFIG, "style intensity must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    style: StyleHints | None = None

    def __post_init__(self) -> None:
        if self.text.strip() == "":
            raise TtsError(TtsErrorKind.CONFIG, "synthesis text must be nonblank")


@dataclass(frozen=True, slots=True)
class TtsAudioFormat:
    sample_rate_hz: int
    channels: int = 1
    dtype: TtsPcmDtype = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise TtsError(TtsErrorKind.CONFIG, "sample_rate_hz must be greater than 0")
        if self.channels != 1:
            raise TtsError(TtsErrorKind.CONFIG, "TTS audio chunks must be mono")
        if self.dtype != "float32":
            raise TtsError(TtsErrorKind.CONFIG, "TTS audio chunks must be float32")


@dataclass(frozen=True, slots=True)
class TTSAudioChunk:
    samples: NDArray[np.float32]
    audio_format: TtsAudioFormat
    sequence: int
    start_seconds: float

    @classmethod
    def from_samples(cls, samples: NDArray[np.float32], audio_format: TtsAudioFormat, sequence: int, start_seconds: float) -> "TTSAudioChunk":
        if samples.ndim != 1:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio samples must be rank 1")
        if samples.size == 0:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio samples must be nonempty")
        if samples.dtype != np.float32:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio samples must be float32")
        if not samples.flags.c_contiguous:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio samples must be contiguous")
        if not np.isfinite(samples).all():
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio samples must be finite")
        if sequence < 0:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio sequence must be nonnegative")
        if start_seconds < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS audio start_seconds must be nonnegative")
        return cls(samples=samples, audio_format=audio_format, sequence=sequence, start_seconds=start_seconds)

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.audio_format.sample_rate_hz


@dataclass(frozen=True, slots=True)
class SynthesisMetrics:
    request_started_at_seconds: float
    time_to_first_audio_seconds: float | None
    synthesis_seconds: float
    generated_audio_seconds: float
    real_time_factor: float
    chunk_count: int
    received_byte_count: int = 0
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if self.request_started_at_seconds < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "request_started_at_seconds must be nonnegative")
        if self.time_to_first_audio_seconds is not None and self.time_to_first_audio_seconds < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "time_to_first_audio_seconds must be nonnegative")
        if self.synthesis_seconds < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "synthesis_seconds must be nonnegative")
        if self.generated_audio_seconds < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "generated_audio_seconds must be nonnegative")
        if self.real_time_factor < 0.0:
            raise TtsError(TtsErrorKind.PROVIDER, "real_time_factor must be nonnegative")
        if self.chunk_count < 0:
            raise TtsError(TtsErrorKind.PROVIDER, "chunk_count must be nonnegative")
        if self.received_byte_count < 0:
            raise TtsError(TtsErrorKind.PROVIDER, "received_byte_count must be nonnegative")
        if self.provider is not None and self.provider.strip() == "":
            raise TtsError(TtsErrorKind.PROVIDER, "provider must be nonblank when set")
        if self.model is not None and self.model.strip() == "":
            raise TtsError(TtsErrorKind.PROVIDER, "model must be nonblank when set")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    request: SynthesisRequest
    status: SynthesisStatus
    metrics: SynthesisMetrics


TtsStreamEvent: TypeAlias = TTSAudioChunk | SynthesisResult


class TTSProvider(Protocol):
    def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]: ...

    async def aclose(self) -> None: ...


def aggregate_audio_duration_seconds(chunks: tuple[TTSAudioChunk, ...]) -> float:
    return sum(chunk.duration_seconds for chunk in chunks)


def real_time_factor(synthesis_seconds: float, generated_audio_seconds: float) -> float:
    if synthesis_seconds < 0.0:
        raise TtsError(TtsErrorKind.PROVIDER, "synthesis_seconds must be nonnegative")
    if generated_audio_seconds < 0.0:
        raise TtsError(TtsErrorKind.PROVIDER, "generated_audio_seconds must be nonnegative")
    if generated_audio_seconds == 0.0:
        return 0.0
    return synthesis_seconds / generated_audio_seconds
