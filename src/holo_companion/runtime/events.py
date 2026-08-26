from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    start_frame_index: int
    detected_frame_index: int
    start_time_ms: float
    type: Literal["speech_started"] = "speech_started"


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    start_frame_index: int
    end_frame_index: int
    detected_frame_index: int
    duration_ms: float
    end_latency_ms: float
    type: Literal["speech_ended"] = "speech_ended"


PublicSpeechEvent: TypeAlias = SpeechStarted | SpeechEnded
