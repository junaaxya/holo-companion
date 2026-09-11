import numpy as np
import pytest

from holo_companion.audio.types import AudioFrame
from holo_companion.runtime.events import SpeechStarted
from holo_companion.stt.base import SttError
from holo_companion.stt.collector import UtteranceCollector
from holo_companion.audio.stream import audio_frame_from_samples
from holo_companion.vad.base import TurnDetector, VadPolicy


def frame(index: int) -> AudioFrame:
    return AudioFrame(samples=np.full((512,), float(index), dtype=np.float32), frame_index=index)


def events_for(probabilities: list[float]) -> list[tuple[AudioFrame, list]]:
    detector = TurnDetector(VadPolicy(min_speech_ms=64, min_silence_ms=64, speech_pad_ms=30, threshold=0.5))
    pairs = []
    for index, probability in enumerate(probabilities):
        audio_frame = frame(index)
        pairs.append((audio_frame, list(detector.process(audio_frame, probability))))
    return pairs


def test_collector_includes_delayed_start_history_and_excludes_debounce_silence() -> None:
    # Given
    collector = UtteranceCollector(history_frames=8)

    # When
    utterances = [collector.process(audio_frame, speech_events) for audio_frame, speech_events in events_for([0.9, 0.9, 0.9, 0.0, 0.0, 0.0])]

    # Then
    utterance = utterances[-1]
    assert utterance is not None
    assert utterance.start_frame_index == 0
    assert utterance.end_frame_index == 3
    assert utterance.samples.shape == (1536,)
    assert np.all(utterance.samples[:512] == 0.0)
    assert np.all(utterance.samples[-512:] == 2.0)


def test_collector_returns_none_for_rejected_transient() -> None:
    # Given
    collector = UtteranceCollector(history_frames=8)

    # When
    utterances = [collector.process(audio_frame, speech_events) for audio_frame, speech_events in events_for([0.9, 0.0, 0.0])]

    # Then
    assert utterances == [None, None, None]


def test_collector_rejects_nonmonotonic_frames() -> None:
    # Given
    collector = UtteranceCollector(history_frames=8)
    collector.process(frame(1), [])

    # When / Then
    with pytest.raises(SttError, match="monotonic"):
        collector.process(frame(1), [])


def test_collector_rejects_event_for_different_frame() -> None:
    # Given
    collector = UtteranceCollector(history_frames=8)

    # When / Then
    with pytest.raises(SttError, match="event/frame mismatch"):
        collector.process(frame(0), [SpeechStarted(start_frame_index=0, detected_frame_index=3, start_time_ms=0.0)])


def test_collector_preserves_owned_frames_after_callback_buffer_reuse_and_reset() -> None:
    # Given
    collector = UtteranceCollector(history_frames=8)
    detector = TurnDetector(VadPolicy(min_speech_ms=64, min_silence_ms=64, speech_pad_ms=30, threshold=0.5))
    source = np.zeros((512, 1), dtype=np.float32)
    utterance = None

    # When
    for index, probability in enumerate([0.9, 0.9, 0.9, 0.0, 0.0, 0.0]):
        source.fill(index / 10.0)
        audio_frame = audio_frame_from_samples(source, index)
        source.fill(-0.77)
        utterance = collector.process(audio_frame, list(detector.process(audio_frame, probability))) or utterance
    assert utterance is not None
    original = utterance.samples.copy()
    collector.process(audio_frame_from_samples(np.full((512,), 0.91, dtype=np.float32), 6), [])

    # Then
    assert utterance.samples.dtype == np.float32
    np.testing.assert_array_equal(utterance.samples, original)
    np.testing.assert_array_equal(utterance.samples[:512], np.zeros(512, dtype=np.float32))
    np.testing.assert_array_equal(utterance.samples[-512:], np.full(512, 0.2, dtype=np.float32))
