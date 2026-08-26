from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from time import monotonic_ns

import numpy as np
import sounddevice
from numpy.typing import NDArray

from holo_companion.audio.types import CAPTURE_FRAME_SAMPLES, AudioCliError, AudioFrame, DeviceIndex


def audio_frame_from_samples(samples: NDArray[np.float32], frame_index: int) -> AudioFrame:
    if samples.shape == (CAPTURE_FRAME_SAMPLES,):
        frame_samples = samples
    elif samples.shape == (CAPTURE_FRAME_SAMPLES, 1):
        frame_samples = samples.reshape(CAPTURE_FRAME_SAMPLES)
    else:
        if len(samples.shape) == 2 and samples.shape[1] != 1:
            raise AudioCliError("audio frame must be mono")
        raise AudioCliError(f"audio frame must contain exactly {CAPTURE_FRAME_SAMPLES} samples")
    if frame_samples.dtype != np.float32:
        frame_samples = frame_samples.astype(np.float32)
    if not np.all(np.isfinite(frame_samples)):
        raise AudioCliError("audio frame contains non-finite samples")
    return AudioFrame(
        samples=np.ascontiguousarray(frame_samples, dtype=np.float32),
        frame_index=frame_index,
        captured_at_ns=monotonic_ns(),
    )


def sounddevice_frame_stream(
    samplerate: int,
    channels: int,
    dtype: str,
    device: DeviceIndex,
    blocksize: int,
) -> Iterator[AudioFrame]:
    frames: Queue[AudioFrame] = Queue(maxsize=8)
    stream_error: Queue[AudioCliError] = Queue(maxsize=1)
    callback_frame_index = _FrameIndex()

    def callback(indata: NDArray[np.float32], received_frames: int, time_info, status) -> None:
        if received_frames != blocksize:
            _put_stream_error(stream_error, AudioCliError(f"audio stream returned {received_frames} frames, expected {blocksize}"))
            return
        try:
            frames.put_nowait(audio_frame_from_samples(indata, frame_index=callback_frame_index.index))
            callback_frame_index.index += 1
        except Full:
            _put_stream_error(stream_error, AudioCliError("audio stream queue overrun"))
        except AudioCliError as error:
            _put_stream_error(stream_error, error)

    try:
        with sounddevice.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            device=int(device),
            blocksize=blocksize,
            callback=callback,
        ):
            while True:
                _raise_stream_error(stream_error)
                try:
                    yield frames.get(timeout=0.1)
                except Empty:
                    _raise_stream_error(stream_error)
    except (sounddevice.PortAudioError, ValueError) as error:
        raise AudioCliError("audio frame stream failed") from error


def _put_stream_error(errors: Queue[AudioCliError], error: AudioCliError) -> None:
    try:
        errors.put_nowait(error)
    except Full:
        return


def _raise_stream_error(errors: Queue[AudioCliError]) -> None:
    try:
        error = errors.get_nowait()
    except Empty:
        return
    raise error


@dataclass(slots=True)
class _FrameIndex:
    index: int = 0
