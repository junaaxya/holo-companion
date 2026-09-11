from dataclasses import dataclass
from enum import StrEnum, unique

from holo_companion.runtime.metrics import TurnTimestamps


@unique
class TurnErrorKind(StrEnum):
    PROVIDER = "provider"
    CANCELLED = "cancelled"
    STALE_OUTPUT = "stale_output"


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    provider: str
    exception_class: str
    message: str
    http_status: int | None = None
    provider_code: str | None = None
    recoverable: bool = True


@dataclass(frozen=True, slots=True)
class InterruptionDiagnostic:
    previous_turn_id: int
    previous_state: str
    new_speech_frame: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class TextSegmentDiagnostic:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class TextTurnDiagnostic:
    llm_text: str
    emitted_segments: tuple[TextSegmentDiagnostic, ...]
    tts_segments: tuple[TextSegmentDiagnostic, ...]
    final_buffer_flushed: bool
    llm_finish_reason: str | None


@dataclass(frozen=True, slots=True)
class TurnError:
    kind: TurnErrorKind
    stage: str
    diagnostic: ProviderDiagnostic | None = None
    interruption: InterruptionDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: int
    generation_id: int
    transcript: str | None
    assistant_text: str
    metrics: TurnTimestamps
    error: TurnError | None = None
    utterance_duration_seconds: float | None = None
    stt_inference_seconds: float | None = None
    utterance_sample_count: int | None = None
    utterance_peak: float | None = None
    utterance_rms: float | None = None
    vad_start_frame_index: int | None = None
    vad_end_frame_index: int | None = None
    text_diagnostic: TextTurnDiagnostic | None = None
