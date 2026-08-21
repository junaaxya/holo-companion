from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, assert_never

import numpy as np
import sounddevice
import soundfile
from numpy.typing import NDArray

from holo_companion.audio.diagnostics import analyze_samples
from holo_companion.audio.types import (
    CAPTURE_CHANNELS,
    CAPTURE_DTYPE,
    CAPTURE_SAMPLE_RATE,
    WAV_FORMAT,
    WAV_SUBTYPE,
    AudioCliError,
    CaptureFormat,
    CaptureRequest,
    CaptureResult,
    DeviceDefaults,
    DeviceIndex,
    DeviceInfo,
    DeviceSelection,
    Frames,
)


class AudioBackend(Protocol):
    def default_input_device(self) -> DeviceInfo: ...

    def default_output_device(self) -> DeviceInfo: ...

    def query_devices(self) -> Sequence[DeviceInfo]: ...

    def query_input_device(self, name_query: str) -> DeviceInfo: ...

    def check_input_settings(self, device: DeviceIndex, samplerate: int, channels: int, dtype: str) -> None: ...

    def record(
        self,
        frames: Frames,
        samplerate: int,
        channels: int,
        dtype: str,
        device: DeviceIndex,
    ) -> NDArray[np.float32]: ...


class WaveWriter(Protocol):
    def write_pcm16(self, path: Path, samples: NDArray[np.float32], samplerate: int) -> None: ...


class SoundDeviceBackend:
    def default_input_device(self) -> DeviceInfo:
        return _default_device(sounddevice.default.device[0], "input")

    def default_output_device(self) -> DeviceInfo:
        return _default_device(sounddevice.default.device[1], "output")

    def query_devices(self) -> Sequence[DeviceInfo]:
        try:
            raw_devices = sounddevice.query_devices()
        except (sounddevice.PortAudioError, ValueError) as error:
            raise AudioCliError("failed to query PortAudio devices") from error
        devices: list[DeviceInfo] = []
        for index, raw_device in enumerate(raw_devices):
            devices.append(_device_from_mapping(DeviceIndex(index), raw_device))
        return devices

    def query_input_device(self, name_query: str) -> DeviceInfo:
        try:
            raw_device = sounddevice.query_devices(name_query, "input")
        except (sounddevice.PortAudioError, ValueError) as error:
            raise AudioCliError(f"input device not found or incompatible: {name_query}") from error
        index = DeviceIndex(int(raw_device["index"]))
        return _device_from_mapping(index, raw_device)

    def check_input_settings(self, device: DeviceIndex, samplerate: int, channels: int, dtype: str) -> None:
        try:
            sounddevice.check_input_settings(
                device=int(device),
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
            )
        except (sounddevice.PortAudioError, ValueError) as error:
            raise AudioCliError(f"input device does not support mono 16000 Hz float32 capture: {device}") from error

    def record(
        self,
        frames: Frames,
        samplerate: int,
        channels: int,
        dtype: str,
        device: DeviceIndex,
    ) -> NDArray[np.float32]:
        try:
            return sounddevice.rec(
                frames=int(frames),
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                blocking=True,
                device=int(device),
            )
        except (sounddevice.PortAudioError, ValueError) as error:
            raise AudioCliError("audio capture failed") from error


class SoundFileWaveWriter:
    def write_pcm16(self, path: Path, samples: NDArray[np.float32], samplerate: int) -> None:
        try:
            soundfile.write(path, samples, samplerate, format=WAV_FORMAT, subtype=WAV_SUBTYPE)
        except soundfile.SoundFileError as error:
            raise AudioCliError(f"failed to write WAV output: {path}") from error


def enumerate_devices(backend: AudioBackend) -> tuple[Sequence[DeviceInfo], DeviceDefaults]:
    devices = backend.query_devices()
    return devices, DeviceDefaults(
        input_selection=DeviceSelection(
            requested="auto",
            resolved=backend.default_input_device(),
        ),
        output_selection=DeviceSelection(
            requested="auto",
            resolved=backend.default_output_device(),
        ),
    )


def record_wav(request: CaptureRequest, backend: AudioBackend, writer: WaveWriter) -> CaptureResult:
    if request.output_path.exists() and not request.overwrite:
        raise AudioCliError(f"output exists; pass --overwrite: {request.output_path}")
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    selection = resolve_input_device(request.requested_input_device, backend)
    audio_format = CaptureFormat()
    frames = Frames(round(float(request.seconds) * int(audio_format.samplerate_hz)))
    backend.check_input_settings(
        selection.resolved.index,
        samplerate=int(audio_format.samplerate_hz),
        channels=audio_format.channels,
        dtype=audio_format.dtype,
    )
    samples = backend.record(
        frames,
        samplerate=int(audio_format.samplerate_hz),
        channels=audio_format.channels,
        dtype=audio_format.dtype,
        device=selection.resolved.index,
    )
    writer.write_pcm16(request.output_path, samples, int(audio_format.samplerate_hz))
    return CaptureResult(
        requested_input_device=selection.requested,
        resolved_input_device=selection.resolved,
        output_path=request.output_path,
        format=audio_format,
        diagnostics=analyze_samples(samples, audio_format.samplerate_hz),
    )


def resolve_input_device(requested: str, backend: AudioBackend) -> DeviceSelection:
    if requested == "auto":
        return DeviceSelection(requested=requested, resolved=backend.default_input_device())
    return DeviceSelection(requested=requested, resolved=backend.query_input_device(requested))


def _default_device(default_value: int | str | None, kind: Literal["input", "output"]) -> DeviceInfo:
    match default_value:
        case None:
            raise AudioCliError(f"no default {kind} device reported by PortAudio")
        case int() as numeric_default if numeric_default < 0:
            raise AudioCliError(f"no default {kind} device reported by PortAudio")
        case int() | str():
            try:
                raw_device = sounddevice.query_devices(default_value, kind)
            except (sounddevice.PortAudioError, ValueError) as error:
                raise AudioCliError(f"default {kind} device not found or incompatible: {default_value}") from error
            device = _device_from_mapping(DeviceIndex(int(raw_device["index"])), raw_device)
            _require_default_kind(device, kind)
            return device
        case unreachable:
            assert_never(unreachable)


def _require_default_kind(device: DeviceInfo, kind: Literal["input", "output"]) -> None:
    match kind:
        case "input":
            if device.max_input_channels < 1:
                raise AudioCliError(f"default device is not an input device: {int(device.index)}")
        case "output":
            if device.max_output_channels < 1:
                raise AudioCliError(f"default output device is not an output device: {int(device.index)}")
        case unreachable:
            assert_never(unreachable)


def _device_from_mapping(index: DeviceIndex, raw_device: sounddevice.DeviceList) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=str(raw_device["name"]),
        max_input_channels=int(raw_device["max_input_channels"]),
        max_output_channels=int(raw_device["max_output_channels"]),
        default_samplerate=float(raw_device["default_samplerate"]),
    )
