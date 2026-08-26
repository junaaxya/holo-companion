from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pytest
import sounddevice
from numpy.typing import NDArray

from holo_companion.audio.capture import SoundDeviceBackend
from holo_companion.audio.stream import audio_frame_from_samples
from holo_companion.audio.types import AudioCliError, DeviceIndex


class InputCallback(Protocol):
    def __call__(self, indata: NDArray[np.float32], frames: int, time_info, status: sounddevice.CallbackFlags) -> None: ...


def test_audio_frame_conversion_accepts_flat_float32_frame_above_unit_amplitude() -> None:
    # Given
    samples = np.full((512,), 1.25, dtype=np.float32)

    # When
    frame = audio_frame_from_samples(samples, frame_index=3)

    # Then
    assert frame.samples.shape == (512,)
    assert frame.samples.dtype == np.float32
    assert frame.samples.flags.c_contiguous is True
    assert frame.frame_index == 3


def test_audio_frame_conversion_accepts_single_channel_column_frame() -> None:
    # Given
    samples = np.zeros((512, 1), dtype=np.float32)

    # When
    frame = audio_frame_from_samples(samples, frame_index=0)

    # Then
    assert frame.samples.shape == (512,)


def test_audio_frame_conversion_rejects_wrong_sample_count() -> None:
    # Given
    samples = np.zeros((511,), dtype=np.float32)

    # When / Then
    with pytest.raises(AudioCliError, match="512 samples"):
        audio_frame_from_samples(samples, frame_index=0)


def test_audio_frame_conversion_rejects_two_channel_layout() -> None:
    # Given
    samples = np.zeros((512, 2), dtype=np.float32)

    # When / Then
    with pytest.raises(AudioCliError, match="mono"):
        audio_frame_from_samples(samples, frame_index=0)


def test_audio_frame_conversion_rejects_non_finite_audio() -> None:
    # Given
    samples = np.zeros((512,), dtype=np.float32)
    samples[10] = np.nan

    # When / Then
    with pytest.raises(AudioCliError, match="non-finite"):
        audio_frame_from_samples(samples, frame_index=0)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeInputStream:
    samplerate: int
    channels: int
    dtype: str
    device: int
    blocksize: int
    callback: InputCallback
    closed: bool = False

    def __enter__(self) -> "FakeInputStream":
        self.callback(np.zeros((512, 1), dtype=np.float32), 512, None, sounddevice.CallbackFlags())
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: BaseException | None) -> None:
        self.closed = True


def test_sounddevice_frame_stream_uses_input_stream_settings_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    streams: list[FakeInputStream] = []

    def input_stream(samplerate: int, channels: int, dtype: str, device: int, blocksize: int, callback: InputCallback) -> FakeInputStream:
        stream = FakeInputStream(samplerate=samplerate, channels=channels, dtype=dtype, device=device, blocksize=blocksize, callback=callback)
        streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", input_stream)

    # When
    frame = next(SoundDeviceBackend().frame_stream(16_000, 1, "float32", DeviceIndex(7), 512))

    # Then
    assert frame.samples.shape == (512,)
    assert streams[0].samplerate == 16_000
    assert streams[0].channels == 1
    assert streams[0].dtype == "float32"
    assert streams[0].device == 7
    assert streams[0].blocksize == 512
    assert streams[0].closed is True


def test_sounddevice_frame_stream_reports_queue_overrun(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    @dataclass(slots=True)  # noqa: MUTABLE_OK
    class OverrunInputStream:
        samplerate: int
        channels: int
        dtype: str
        device: int
        blocksize: int
        callback: InputCallback

        def __enter__(self) -> "OverrunInputStream":
            for _ in range(9):
                self.callback(np.zeros((512, 1), dtype=np.float32), 512, None, sounddevice.CallbackFlags())
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: BaseException | None) -> None:
            return None

    def input_stream(samplerate: int, channels: int, dtype: str, device: int, blocksize: int, callback: InputCallback) -> OverrunInputStream:
        return OverrunInputStream(samplerate=samplerate, channels=channels, dtype=dtype, device=device, blocksize=blocksize, callback=callback)

    monkeypatch.setattr(sounddevice, "InputStream", input_stream)

    # When / Then
    with pytest.raises(AudioCliError, match="overrun"):
        next(SoundDeviceBackend().frame_stream(16_000, 1, "float32", DeviceIndex(7), 512))
