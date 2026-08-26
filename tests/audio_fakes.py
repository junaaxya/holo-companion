from dataclasses import dataclass, field
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from holo_companion.audio.types import AudioCliError, AudioFrame, DeviceIndex, DeviceInfo, Frames


@dataclass(frozen=True, slots=True)
class CheckCall:
    device: DeviceIndex
    samplerate: int
    channels: int
    dtype: str


@dataclass(frozen=True, slots=True)
class RecordCall:
    frames: Frames
    samplerate: int
    channels: int
    dtype: str
    device: DeviceIndex


@dataclass(frozen=True, slots=True)
class StreamCall:
    samplerate: int
    channels: int
    dtype: str
    device: DeviceIndex
    blocksize: int


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeAudioBackend:
    devices: list[DeviceInfo]
    default_input: DeviceIndex
    default_output: DeviceIndex
    samples: NDArray[np.float32]
    incompatible: bool = False
    check_calls: list[CheckCall] = field(default_factory=list)
    record_calls: list[RecordCall] = field(default_factory=list)
    stream_calls: list[StreamCall] = field(default_factory=list)
    stream_frames: list[AudioFrame] = field(default_factory=list)

    def default_input_device(self) -> DeviceInfo:
        matches = [device for device in self.devices if device.index == self.default_input]
        if len(matches) != 1:
            raise AudioCliError(f"default input device index not found: {int(self.default_input)}")
        device = matches[0]
        if device.max_input_channels < 1:
            raise AudioCliError(f"default device is not an input device: {int(self.default_input)}")
        return device

    def default_output_device(self) -> DeviceInfo:
        matches = [device for device in self.devices if device.index == self.default_output]
        if len(matches) != 1:
            raise AudioCliError(f"default output device index not found: {int(self.default_output)}")
        device = matches[0]
        if device.max_output_channels < 1:
            raise AudioCliError(f"default output device is not an output device: {int(self.default_output)}")
        return device

    def query_devices(self) -> list[DeviceInfo]:
        return self.devices

    def query_input_device(self, name_query: str) -> DeviceInfo:
        matches = [device for device in self.devices if name_query.lower() in device.name.lower()]
        if len(matches) != 1:
            raise AudioCliError(f"input device not found or incompatible: {name_query}")
        device = matches[0]
        if device.max_input_channels < 1:
            raise AudioCliError(f"input device not found or incompatible: {name_query}")
        return device

    def check_input_settings(self, device: DeviceIndex, samplerate: int, channels: int, dtype: str) -> None:
        self.check_calls.append(CheckCall(device=device, samplerate=samplerate, channels=channels, dtype=dtype))
        if self.incompatible:
            raise AudioCliError("input device does not support mono 16000 Hz float32 capture")

    def record(
        self,
        frames: Frames,
        samplerate: int,
        channels: int,
        dtype: str,
        device: DeviceIndex,
    ) -> NDArray[np.float32]:
        self.record_calls.append(
            RecordCall(frames=frames, samplerate=samplerate, channels=channels, dtype=dtype, device=device),
        )
        return self.samples

    def frame_stream(
        self,
        samplerate: int,
        channels: int,
        dtype: str,
        device: DeviceIndex,
        blocksize: int,
    ) -> Iterator[AudioFrame]:
        self.stream_calls.append(StreamCall(samplerate=samplerate, channels=channels, dtype=dtype, device=device, blocksize=blocksize))
        yield from self.stream_frames


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeWaveWriter:
    writes: list[tuple[Path, NDArray[np.float32], int]] = field(default_factory=list)

    def write_pcm16(self, path: Path, samples: NDArray[np.float32], samplerate: int) -> None:
        self.writes.append((path, samples, samplerate))


def fake_backend(
    samples: NDArray[np.float32] | None = None,
    incompatible: bool = False,
    default_output: DeviceIndex = DeviceIndex(0),
) -> FakeAudioBackend:
    default_samples = np.array([[0.0], [0.01], [-0.01], [0.02]], dtype=np.float32)
    return FakeAudioBackend(
        devices=[
            DeviceInfo(DeviceIndex(0), "Built-in Speaker", 0, 2, 48_000.0),
            DeviceInfo(DeviceIndex(1), "USB Microphone", 1, 0, 48_000.0),
        ],
        default_input=DeviceIndex(1),
        default_output=default_output,
        samples=default_samples if samples is None else samples,
        incompatible=incompatible,
    )
