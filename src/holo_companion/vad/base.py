from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from holo_companion.audio.types import CAPTURE_FRAME_MS, AudioCliError, AudioFrame
from holo_companion.runtime.events import PublicSpeechEvent, SpeechEnded, SpeechStarted


@dataclass(frozen=True, slots=True)
class VadPolicy:
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    threshold: float

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise AudioCliError("VAD threshold must be greater than 0 and less than 1")
        if self.min_speech_ms <= 0:
            raise AudioCliError("minimum speech duration must be greater than 0")
        if self.min_silence_ms <= 0:
            raise AudioCliError("minimum silence duration must be greater than 0")
        if self.speech_pad_ms < 0:
            raise AudioCliError("speech padding must be at least 0")


class VadProvider(Protocol):
    def process(self, frame: AudioFrame) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VadDiagnostics:
    processed_frames: int
    speech_started_count: int
    speech_ended_count: int
    rejected_speech_candidates: int
    last_end_latency_ms: float | None


@dataclass(slots=True)  # noqa: MUTABLE_OK
class TurnDetector:
    policy: VadPolicy
    candidate_start_frame_index: int | None = None
    active_start_frame_index: int | None = None
    silence_start_frame_index: int | None = None
    speech_started_count: int = 0
    speech_ended_count: int = 0
    rejected_speech_candidates: int = 0
    processed_frames: int = 0
    last_end_latency_ms: float | None = None

    def process(self, frame: AudioFrame, speech_probability: float) -> Iterator[PublicSpeechEvent]:
        self.processed_frames += 1
        is_speech = speech_probability >= self.policy.threshold
        if is_speech:
            self.silence_start_frame_index = None
            if self.candidate_start_frame_index is None and self.active_start_frame_index is None:
                self.candidate_start_frame_index = frame.frame_index
            yield from self._accept_candidate(frame.frame_index)
            return
        if self.active_start_frame_index is None:
            if self.candidate_start_frame_index is not None:
                self.candidate_start_frame_index = None
                self.rejected_speech_candidates += 1
            return
        if self.silence_start_frame_index is None:
            self.silence_start_frame_index = frame.frame_index
        yield from self._finish_turn(frame.frame_index)

    def reset(self) -> None:
        self.candidate_start_frame_index = None
        self.active_start_frame_index = None
        self.silence_start_frame_index = None
        self.speech_started_count = 0
        self.speech_ended_count = 0
        self.rejected_speech_candidates = 0
        self.processed_frames = 0
        self.last_end_latency_ms = None

    @property
    def diagnostics(self) -> VadDiagnostics:
        return VadDiagnostics(
            processed_frames=self.processed_frames,
            speech_started_count=self.speech_started_count,
            speech_ended_count=self.speech_ended_count,
            rejected_speech_candidates=self.rejected_speech_candidates,
            last_end_latency_ms=self.last_end_latency_ms,
        )

    def _accept_candidate(self, observed_frame_index: int) -> Iterator[SpeechStarted]:
        if self.candidate_start_frame_index is None:
            return
        elapsed_ms = _frame_time_ms(observed_frame_index - self.candidate_start_frame_index)
        if elapsed_ms < self.policy.min_speech_ms:
            return
        start_frame_index = self.candidate_start_frame_index
        self.candidate_start_frame_index = None
        self.active_start_frame_index = start_frame_index
        self.speech_started_count += 1
        yield SpeechStarted(
            start_frame_index=start_frame_index,
            detected_frame_index=observed_frame_index,
            start_time_ms=_frame_time_ms(start_frame_index),
        )

    def _finish_turn(self, observed_frame_index: int) -> Iterator[SpeechEnded]:
        if self.active_start_frame_index is None or self.silence_start_frame_index is None:
            return
        silence_ms = _frame_time_ms(observed_frame_index - self.silence_start_frame_index)
        if silence_ms < self.policy.min_silence_ms:
            return
        start_frame_index = self.active_start_frame_index
        end_frame_index = self.silence_start_frame_index
        end_latency_ms = _frame_time_ms(observed_frame_index - end_frame_index)
        self.active_start_frame_index = None
        self.silence_start_frame_index = None
        self.last_end_latency_ms = end_latency_ms
        self.speech_ended_count += 1
        yield SpeechEnded(
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            detected_frame_index=observed_frame_index,
            duration_ms=_frame_time_ms(end_frame_index - start_frame_index),
            end_latency_ms=end_latency_ms,
        )


def _frame_time_ms(frame_count: int) -> float:
    return frame_count * CAPTURE_FRAME_MS
