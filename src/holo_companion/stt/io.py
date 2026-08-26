from pathlib import Path

import numpy as np
import soundfile

from holo_companion.audio.types import CAPTURE_SAMPLE_RATE
from holo_companion.stt.base import SttError, Utterance


def load_canonical_wav(path: Path) -> Utterance:
    try:
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    except soundfile.SoundFileError as error:
        raise SttError(f"failed to read WAV: {path}") from error
    if sample_rate != int(CAPTURE_SAMPLE_RATE) or samples.ndim != 1:
        raise SttError("STT WAV input must be mono 16000 Hz; resample/downmix explicitly before this command")
    return Utterance.from_samples(np.ascontiguousarray(samples, dtype=np.float32))
