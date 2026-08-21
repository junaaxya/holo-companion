import numpy as np

from holo_companion.audio.diagnostics import analyze_samples
from holo_companion.audio.types import CAPTURE_SAMPLE_RATE, Frames


def test_diagnostics_are_deterministic_when_samples_include_low_volume_and_clipping() -> None:
    # Given
    samples = np.array([[0.0], [0.01], [-0.01], [1.0], [-1.0]], dtype=np.float32)

    # When
    diagnostics = analyze_samples(samples, CAPTURE_SAMPLE_RATE)

    # Then
    assert diagnostics.frames == Frames(5)
    assert diagnostics.duration_seconds == 5 / 16_000
    assert diagnostics.peak_abs_amplitude == 1.0
    assert diagnostics.clipping_samples == 2
    assert diagnostics.clipping is True
    assert diagnostics.low_volume_status == "ok"


def test_diagnostics_report_low_volume_when_peak_is_below_threshold() -> None:
    # Given
    samples = np.array([[0.0], [0.01], [-0.01]], dtype=np.float32)

    # When
    diagnostics = analyze_samples(samples, CAPTURE_SAMPLE_RATE)

    # Then
    assert diagnostics.low_volume_threshold == 0.02
    assert diagnostics.low_volume_status == "low_volume"
