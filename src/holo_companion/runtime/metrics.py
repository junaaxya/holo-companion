import time
from dataclasses import dataclass
from typing import Protocol


class MonotonicClock(Protocol):
    def now(self) -> float: ...


@dataclass(slots=True)
class SystemMonotonicClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class ManualClock:
    current_seconds: float = 0.0

    def now(self) -> float:
        return self.current_seconds

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("manual clock cannot move backward")
        self.current_seconds += seconds


@dataclass(frozen=True, slots=True)
class TurnTimestamps:
    speech_end_detected: float | None = None
    stt_start: float | None = None
    stt_invocation_start: float | None = None
    stt_complete: float | None = None
    llm_invocation_start: float | None = None
    llm_request_start: float | None = None
    first_llm_raw_delta: float | None = None
    first_llm_token: float | None = None
    tts_request_start: float | None = None
    first_tts_audio: float | None = None
    first_playback_audio: float | None = None

    @property
    def speech_end_to_transcript_seconds(self) -> float | None:
        return _delta(self.speech_end_detected, self.stt_complete)

    @property
    def stt_wait_seconds(self) -> float | None:
        return _delta(self.speech_end_detected, self.stt_invocation_start)

    @property
    def stt_invocation_seconds(self) -> float | None:
        return _delta(self.stt_invocation_start, self.stt_complete)

    @property
    def speech_end_to_first_token_seconds(self) -> float | None:
        return _delta(self.speech_end_detected, self.first_llm_token)

    @property
    def llm_invocation_to_provider_request_seconds(self) -> float | None:
        return _delta(self.llm_invocation_start, self.llm_request_start)

    @property
    def llm_provider_request_to_first_raw_delta_seconds(self) -> float | None:
        return _delta(self.llm_request_start, self.first_llm_raw_delta)

    @property
    def llm_provider_request_to_first_usable_delta_seconds(self) -> float | None:
        return _delta(self.llm_request_start, self.first_llm_token)

    @property
    def speech_end_to_first_audio_seconds(self) -> float | None:
        return _delta(self.speech_end_detected, self.first_tts_audio)

    @property
    def speech_end_to_first_playback_seconds(self) -> float | None:
        return _delta(self.speech_end_detected, self.first_playback_audio)


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start
