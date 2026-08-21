from typing import Final

import numpy as np
from numpy.typing import NDArray

from holo_companion.audio.types import (
    LOW_VOLUME_PEAK_THRESHOLD,
    AudioDiagnostics,
    Frames,
    SampleRateHz,
)

LOW_VOLUME_STATUS_OK: Final = "ok"
LOW_VOLUME_STATUS_LOW: Final = "low_volume"


def analyze_samples(samples: NDArray[np.float32], samplerate_hz: SampleRateHz) -> AudioDiagnostics:
    flattened = samples.reshape(-1)
    frames = Frames(int(samples.shape[0]))
    duration_seconds = int(frames) / int(samplerate_hz)
    absolute = np.abs(flattened)
    peak = float(np.max(absolute, initial=np.float32(0.0)))
    rms = float(np.sqrt(np.mean(np.square(flattened), dtype=np.float64))) if flattened.size > 0 else 0.0
    clipping_samples = int(np.count_nonzero(absolute >= 1.0))
    low_volume_status = LOW_VOLUME_STATUS_LOW if peak < LOW_VOLUME_PEAK_THRESHOLD else LOW_VOLUME_STATUS_OK
    return AudioDiagnostics(
        frames=frames,
        duration_seconds=duration_seconds,
        peak_abs_amplitude=peak,
        rms=rms,
        clipping_samples=clipping_samples,
        clipping=clipping_samples > 0,
        low_volume_threshold=LOW_VOLUME_PEAK_THRESHOLD,
        low_volume_status=low_volume_status,
    )
