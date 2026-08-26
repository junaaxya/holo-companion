from collections import deque
from dataclasses import dataclass, field
from typing import assert_never

import numpy as np

from holo_companion.audio.types import AudioFrame
from holo_companion.runtime.events import PublicSpeechEvent, SpeechEnded, SpeechStarted
from holo_companion.stt.base import SttError, Utterance


@dataclass(slots=True)  # noqa: MUTABLE_OK
class UtteranceCollector:
    history_frames: int
    history: deque[AudioFrame] = field(init=False)
    active_frames: list[AudioFrame] | None = None
    active_start_frame_index: int | None = None
    last_frame_index: int | None = None

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise SttError("utterance history must keep at least one frame")
        self.history = deque(maxlen=self.history_frames)

    def process(self, frame: AudioFrame, events: list[PublicSpeechEvent]) -> Utterance | None:
        self._require_next_frame(frame)
        self.history.append(frame)
        utterance: Utterance | None = None
        for event in events:
            self._require_event_matches_frame(frame, event)
            match event:
                case SpeechStarted() as started:
                    self._start(started)
                case SpeechEnded() as ended:
                    utterance = self._end(ended)
                case unreachable:
                    assert_never(unreachable)
        if self.active_frames is not None and self.active_frames[-1].frame_index != frame.frame_index:
            self.active_frames.append(frame)
        return utterance

    def _require_next_frame(self, frame: AudioFrame) -> None:
        if self.last_frame_index is not None and frame.frame_index <= self.last_frame_index:
            raise SttError("audio frame indexes must be monotonic")
        self.last_frame_index = frame.frame_index

    def _require_event_matches_frame(self, frame: AudioFrame, event: PublicSpeechEvent) -> None:
        match event:
            case SpeechStarted(detected_frame_index=detected_frame_index) | SpeechEnded(detected_frame_index=detected_frame_index):
                if detected_frame_index != frame.frame_index:
                    raise SttError("speech event/frame mismatch")
            case unreachable:
                assert_never(unreachable)

    def _start(self, event: SpeechStarted) -> None:
        frames = [frame for frame in self.history if frame.frame_index >= event.start_frame_index]
        if not frames or frames[0].frame_index != event.start_frame_index:
            raise SttError("speech start history is no longer available")
        self.active_frames = frames
        self.active_start_frame_index = event.start_frame_index

    def _end(self, event: SpeechEnded) -> Utterance | None:
        if self.active_frames is None or self.active_start_frame_index is None:
            return None
        frames = [frame for frame in self.active_frames if frame.frame_index < event.end_frame_index]
        self.active_frames = None
        self.active_start_frame_index = None
        if not frames:
            return None
        samples = np.ascontiguousarray(np.concatenate([frame.samples for frame in frames]), dtype=np.float32)
        return Utterance.from_samples(samples, start_frame_index=event.start_frame_index, end_frame_index=event.end_frame_index)
