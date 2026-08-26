import numpy as np
import pytest

from holo_companion.audio.types import AudioCliError, AudioFrame
from holo_companion.runtime.events import SpeechEnded, SpeechStarted
from holo_companion.vad.base import TurnDetector, VadPolicy


def frame(index: int) -> AudioFrame:
    return AudioFrame(samples=np.zeros((512,), dtype=np.float32), frame_index=index)


def policy(*, min_speech_ms: int = 250, min_silence_ms: int = 500) -> VadPolicy:
    return VadPolicy(min_speech_ms=min_speech_ms, min_silence_ms=min_silence_ms, speech_pad_ms=30, threshold=0.5)


def test_detector_delays_start_until_minimum_speech_duration() -> None:
    # Given
    detector = TurnDetector(policy())

    # When
    early_events = [list(detector.process(frame(index), 0.9)) for index in range(8)]
    accepted_events = list(detector.process(frame(8), 0.9))

    # Then
    assert all(events == [] for events in early_events)
    assert accepted_events == [SpeechStarted(start_frame_index=0, detected_frame_index=8, start_time_ms=0.0)]


def test_detector_emits_end_after_configured_silence() -> None:
    # Given
    detector = TurnDetector(policy())
    for index in range(9):
        list(detector.process(frame(index), 0.9))

    # When
    early_end_events = [list(detector.process(frame(index), 0.0)) for index in range(9, 25)]
    ended_events = list(detector.process(frame(25), 0.0))

    # Then
    assert all(events == [] for events in early_end_events)
    assert ended_events == [SpeechEnded(start_frame_index=0, end_frame_index=9, detected_frame_index=25, duration_ms=288.0, end_latency_ms=512.0)]


def test_detector_keeps_turn_open_during_short_pause() -> None:
    # Given
    detector = TurnDetector(policy())
    for index in range(9):
        list(detector.process(frame(index), 0.9))

    # When
    pause_events = [list(detector.process(frame(index), 0.0)) for index in range(9, 19)]
    resumed_events = list(detector.process(frame(19), 0.9))

    # Then
    assert all(events == [] for events in pause_events)
    assert resumed_events == []
    assert detector.diagnostics.speech_ended_count == 0


def test_detector_rejects_192ms_transient_when_minimum_is_500ms() -> None:
    # Given
    detector = TurnDetector(policy(min_speech_ms=500, min_silence_ms=700))

    # When
    speech_events = [list(detector.process(frame(index), 0.9)) for index in range(6)]
    silence_events = list(detector.process(frame(6), 0.0))

    # Then
    assert all(events == [] for events in speech_events)
    assert silence_events == []
    assert detector.diagnostics.rejected_speech_candidates == 1
    assert detector.diagnostics.speech_started_count == 0


def test_detector_accepts_repeated_turns() -> None:
    # Given
    detector = TurnDetector(policy(min_speech_ms=64, min_silence_ms=64))

    # When
    events = []
    for index, probability in enumerate([0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0]):
        events.extend(detector.process(frame(index), probability))

    # Then
    assert [event.type for event in events] == ["speech_started", "speech_ended", "speech_started", "speech_ended"]
    assert detector.diagnostics.speech_started_count == 2
    assert detector.diagnostics.speech_ended_count == 2


def test_detector_reset_clears_state_and_diagnostics() -> None:
    # Given
    detector = TurnDetector(policy())
    list(detector.process(frame(0), 0.9))

    # When
    detector.reset()

    # Then
    assert detector.diagnostics.processed_frames == 0
    assert detector.diagnostics.speech_started_count == 0
    assert detector.diagnostics.rejected_speech_candidates == 0


def test_vad_policy_rejects_invalid_values() -> None:
    # Given / When / Then
    with pytest.raises(AudioCliError):
        VadPolicy(min_speech_ms=250, min_silence_ms=500, speech_pad_ms=30, threshold=1.0)
