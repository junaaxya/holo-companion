import io
from dataclasses import dataclass

import numpy as np
import pytest
import sounddevice

from holo_companion.audio.capture import SoundDeviceBackend, enumerate_devices
from holo_companion.audio.types import AudioCliError, DeviceIndex, Frames
from holo_companion.cli import run_cli

from audio_fakes import FakeWaveWriter


DeviceMapping = dict[str, int | float | str]


@dataclass(frozen=True, slots=True)
class DefaultDevice:
    device: tuple[int | str | None, int | str | None]


def test_default_input_device_uses_sounddevice_query_when_default_is_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=("USB Microphone", "Built-in Speaker")))

    def query_devices(device: int | str | None = None, kind: str | None = None) -> dict[str, int | float | str]:
        assert device == "USB Microphone"
        assert kind == "input"
        return {"index": 7, "name": "USB Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48_000.0}

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When
    device = SoundDeviceBackend().default_input_device()

    # Then
    assert device.index == DeviceIndex(7)


def test_default_output_device_uses_sounddevice_query_when_default_is_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=("USB Microphone", "Built-in Speaker")))

    def query_devices(device: int | str | None = None, kind: str | None = None) -> dict[str, int | float | str]:
        assert device == "Built-in Speaker"
        assert kind == "output"
        return {"index": 3, "name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48_000.0}

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When
    device = SoundDeviceBackend().default_output_device()

    # Then
    assert device.index == DeviceIndex(3)


def test_default_devices_use_sounddevice_query_when_defaults_are_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=(7, 3)))

    def query_devices(device: int | str | None = None, kind: str | None = None) -> DeviceMapping:
        if device == 7 and kind == "input":
            return {"index": 7, "name": "USB Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48_000.0}
        if device == 3 and kind == "output":
            return {"index": 3, "name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48_000.0}
        raise ValueError("unexpected query")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When
    backend = SoundDeviceBackend()

    # Then
    assert backend.default_input_device().index == DeviceIndex(7)
    assert backend.default_output_device().index == DeviceIndex(3)


def test_enumerate_devices_reports_named_default_observed_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=("USB Microphone", "Built-in Speaker")))

    input_device: DeviceMapping = {"index": 7, "name": "USB Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48_000.0}
    output_device: DeviceMapping = {"index": 3, "name": "Built-in Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48_000.0}

    def query_devices(device: int | str | None = None, kind: str | None = None) -> list[DeviceMapping] | DeviceMapping:
        if device is None and kind is None:
            return [output_device, input_device]
        if device == "USB Microphone" and kind == "input":
            return input_device
        if device == "Built-in Speaker" and kind == "output":
            return output_device
        raise ValueError("unexpected query")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When
    devices, defaults = enumerate_devices(SoundDeviceBackend())

    # Then
    assert defaults.input_selection.resolved.index == DeviceIndex(7)
    assert defaults.output_selection.resolved.index == DeviceIndex(3)
    assert [device.index for device in devices] == [DeviceIndex(0), DeviceIndex(1)]


def test_default_input_device_raises_audio_cli_error_when_default_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=(None, 0)))

    # When / Then
    with pytest.raises(AudioCliError, match="no default input device"):
        SoundDeviceBackend().default_input_device()


def test_default_output_device_raises_audio_cli_error_when_default_is_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=(0, -1)))

    # When / Then
    with pytest.raises(AudioCliError, match="no default output device"):
        SoundDeviceBackend().default_output_device()


def test_default_input_device_converts_query_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=("USB Microphone", 0)))

    def query_devices(device: int | str | None = None, kind: str | None = None) -> DeviceMapping:
        raise ValueError("missing default")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When / Then
    with pytest.raises(AudioCliError, match="default input device not found or incompatible"):
        SoundDeviceBackend().default_input_device()


def test_default_output_device_converts_query_portaudio_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sounddevice, "default", DefaultDevice(device=(0, "Built-in Speaker")))

    def query_devices(device: int | str | None = None, kind: str | None = None) -> DeviceMapping:
        raise sounddevice.PortAudioError("missing default")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When / Then
    with pytest.raises(AudioCliError, match="default output device not found or incompatible"):
        SoundDeviceBackend().default_output_device()


def test_query_input_device_converts_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    def query_devices(device: str, kind: str) -> dict[str, int | float | str]:
        raise ValueError("ambiguous device")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When / Then
    with pytest.raises(AudioCliError, match="input device not found or incompatible: usb"):
        SoundDeviceBackend().query_input_device("usb")


def test_devices_command_converts_query_failure_to_exit_2_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    stderr = io.StringIO()

    def query_devices(device: int | str | None = None, kind: str | None = None) -> list[DeviceMapping] | DeviceMapping:
        raise sounddevice.PortAudioError("host API unavailable")

    monkeypatch.setattr(sounddevice, "query_devices", query_devices)

    # When
    code = run_cli(["devices"], SoundDeviceBackend(), FakeWaveWriter(), io.StringIO(), stderr)

    # Then
    assert code == 2
    assert "failed to query PortAudio devices" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_check_input_settings_converts_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    def check_input_settings(device: int, samplerate: int, channels: int, dtype: str) -> None:
        raise ValueError("invalid settings")

    monkeypatch.setattr(sounddevice, "check_input_settings", check_input_settings)

    # When / Then
    with pytest.raises(AudioCliError, match="does not support"):
        SoundDeviceBackend().check_input_settings(DeviceIndex(1), samplerate=16_000, channels=1, dtype="float32")


def test_record_converts_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    def rec(frames: int, samplerate: int, channels: int, dtype: str, blocking: bool, device: int) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        raise ValueError("invalid record args")

    monkeypatch.setattr(sounddevice, "rec", rec)

    # When / Then
    with pytest.raises(AudioCliError, match="audio capture failed"):
        SoundDeviceBackend().record(Frames(160), samplerate=16_000, channels=1, dtype="float32", device=DeviceIndex(1))
