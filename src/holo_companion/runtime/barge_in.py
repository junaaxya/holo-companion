from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from holo_companion.audio.types import CAPTURE_FRAME_MS, CAPTURE_SAMPLE_RATE, AudioFrame
from holo_companion.runtime.events import SpeechStarted

DEFAULT_BARGE_IN_SUSTAINED_MS = 300
DEFAULT_BARGE_IN_CONFIDENCE_MARGIN = 0.15
DEFAULT_BARGE_IN_RMS_RATIO = 3.0
DEFAULT_ECHO_CORRELATION_THRESHOLD = 0.85
DEFAULT_ECHO_REFERENCE_SECONDS = 1.0
DEFAULT_AMBIENT_HISTORY_FRAMES = 64
ECHO_WINDOW_SAMPLES = 512

GateRejection = Literal["noise", "short", "echo"]


@dataclass(slots=True)
class AdaptiveBargeInGate:
    sustained_ms: int = DEFAULT_BARGE_IN_SUSTAINED_MS
    confidence_margin: float = DEFAULT_BARGE_IN_CONFIDENCE_MARGIN
    rms_ratio: float = DEFAULT_BARGE_IN_RMS_RATIO
    noise_floor: float | None = None
    ambient_history: deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_AMBIENT_HISTORY_FRAMES))
    pending: SpeechStarted | None = None
    echo_correlation_threshold: float = DEFAULT_ECHO_CORRELATION_THRESHOLD
    echo_reference: deque[NDArray[np.float32]] = field(default_factory=deque)
    candidate_samples: list[NDArray[np.float32]] = field(default_factory=list)
    last_echo_correlation: float | None = None
    last_echo_best_delay_ms: float | None = None
    last_playback_reference_rms: float | None = None
    last_candidate_rms: float | None = None
    last_ambient_noise_rms: float | None = None
    last_rms_ratio: float | None = None
    last_vad_confidence: float | None = None
    last_reject_reason: GateRejection | None = None

    def observe_verified_nonspeech(self, frame: AudioFrame) -> None:
        self.ambient_history.append(_rms(frame))
        self.noise_floor = float(np.median(tuple(self.ambient_history)))

    def begin(self, event: SpeechStarted, frame: AudioFrame, probability: float, threshold: float) -> tuple[SpeechStarted | None, GateRejection | None]:
        self.pending = event
        self.candidate_samples = [frame.samples]
        return self.advance(frame, probability, threshold)

    def advance(self, frame: AudioFrame, probability: float, threshold: float) -> tuple[SpeechStarted | None, GateRejection | None]:
        pending = self.pending
        if pending is None:
            return None, None
        if self.candidate_samples[-1] is not frame.samples:
            self.candidate_samples.append(frame.samples)
        if probability < threshold:
            return None, None
        if (frame.frame_index - pending.start_frame_index) * CAPTURE_FRAME_MS < self.sustained_ms:
            return None, None
        candidate = np.concatenate(self.candidate_samples)
        self.last_candidate_rms = _samples_rms(candidate)
        self.last_ambient_noise_rms = self.noise_floor
        self.last_rms_ratio = None if self.noise_floor is None else self.last_candidate_rms / max(self.noise_floor, np.finfo(np.float32).eps)
        self.last_vad_confidence = probability
        reference = self._reference_samples()
        self.last_playback_reference_rms = _samples_rms(reference) if reference.size else None
        self.last_echo_correlation, self.last_echo_best_delay_ms = _echo_match(candidate, reference)
        if self.last_echo_correlation >= self.echo_correlation_threshold:
            return self._reject("echo")
        if probability < threshold + self.confidence_margin or self.noise_floor is None or self.last_rms_ratio < self.rms_ratio:
            return self._reject("noise")
        self.pending = None
        self.candidate_samples = []
        self.last_reject_reason = None
        return SpeechStarted(
            start_frame_index=pending.start_frame_index,
            detected_frame_index=frame.frame_index,
            start_time_ms=pending.start_time_ms,
        ), None

    def reject_short(self) -> bool:
        if self.pending is None:
            return False
        self.pending = None
        self.candidate_samples = []
        self.last_reject_reason = "short"
        return True

    def observe_playback(self, samples: NDArray[np.float32], sample_rate_hz: int) -> None:
        reference = _resample(samples, sample_rate_hz, int(CAPTURE_SAMPLE_RATE))
        self.echo_reference.append(reference)
        maximum = int(DEFAULT_ECHO_REFERENCE_SECONDS * int(CAPTURE_SAMPLE_RATE))
        while sum(chunk.size for chunk in self.echo_reference) > maximum:
            self.echo_reference.popleft()

    def _reference_samples(self) -> NDArray[np.float32]:
        if not self.echo_reference:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(tuple(self.echo_reference))

    def _reject(self, reason: GateRejection) -> tuple[None, GateRejection]:
        self.pending = None
        self.candidate_samples = []
        self.last_reject_reason = reason
        return None, reason


def _rms(frame: AudioFrame) -> float:
    return _samples_rms(frame.samples)


def _samples_rms(samples: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _resample(samples: NDArray[np.float32], source_rate_hz: int, target_rate_hz: int) -> NDArray[np.float32]:
    if source_rate_hz == target_rate_hz:
        return samples
    target_size = round(samples.size * target_rate_hz / source_rate_hz)
    positions = np.linspace(0, samples.size - 1, target_size)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def _echo_match(candidate: NDArray[np.float32], reference: NDArray[np.float32]) -> tuple[float, float | None]:
    if candidate.size == 0 or reference.size < candidate.size:
        return 0.0, None
    best = 0.0
    best_offset = 0
    for offset in range(reference.size - candidate.size + 1):
        correlations = [_correlation(candidate[start : start + ECHO_WINDOW_SAMPLES], reference[offset + start : offset + start + ECHO_WINDOW_SAMPLES]) for start in range(0, candidate.size - ECHO_WINDOW_SAMPLES + 1, ECHO_WINDOW_SAMPLES)]
        if correlations:
            score = float(np.median(correlations))
            if score > best:
                best = score
                best_offset = offset
    delay_ms = (reference.size - best_offset - candidate.size) / int(CAPTURE_SAMPLE_RATE) * 1000
    return best, delay_ms


def _correlation(candidate: NDArray[np.float32], reference: NDArray[np.float32]) -> float:
    candidate_centered = candidate - candidate.mean()
    reference_centered = reference - reference.mean()
    denominator = np.linalg.norm(candidate_centered) * np.linalg.norm(reference_centered)
    return 0.0 if denominator == 0 else abs(float(np.dot(candidate_centered, reference_centered) / denominator))
