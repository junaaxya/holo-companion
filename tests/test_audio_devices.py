import io
import json

import pytest

from holo_companion.audio.capture import enumerate_devices, resolve_input_device
from holo_companion.audio.types import AudioCliError, DeviceIndex
from holo_companion.cli import run_cli

from audio_fakes import FakeWaveWriter, fake_backend


def test_auto_resolution_when_default_input_exists() -> None:
    # Given
    backend = fake_backend()

    # When
    selection = resolve_input_device("auto", backend)

    # Then
    assert selection.requested == "auto"
    assert selection.resolved.name == "USB Microphone"


def test_name_resolution_when_query_matches_input_device() -> None:
    # Given
    backend = fake_backend()

    # When
    selection = resolve_input_device("usb", backend)

    # Then
    assert selection.requested == "usb"
    assert selection.resolved.index == DeviceIndex(1)


def test_name_resolution_fails_when_query_matches_output_only() -> None:
    # Given
    backend = fake_backend()

    # When / Then
    with pytest.raises(AudioCliError, match="input device not found"):
        resolve_input_device("speaker", backend)


def test_enumerate_devices_reports_runtime_default_input_and_output() -> None:
    # Given
    backend = fake_backend()

    # When
    devices, defaults = enumerate_devices(backend)

    # Then
    assert defaults.input_selection.requested == "auto"
    assert defaults.input_selection.resolved.index == DeviceIndex(1)
    assert defaults.output_selection.requested == "auto"
    assert defaults.output_selection.resolved.index == DeviceIndex(0)
    assert [device.index for device in devices] == [DeviceIndex(0), DeviceIndex(1)]


def test_enumerate_devices_rejects_default_output_without_output_channels() -> None:
    # Given
    backend = fake_backend(default_output=DeviceIndex(1))

    # When / Then
    with pytest.raises(AudioCliError, match="default output device is not an output device"):
        enumerate_devices(backend)


def test_devices_command_reports_input_output_auto_resolution_and_observed_indexes() -> None:
    # Given
    stdout = io.StringIO()

    # When
    code = run_cli(["devices"], fake_backend(), FakeWaveWriter(), stdout, io.StringIO())

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["input_device_request"] == "auto"
    assert payload["input_device_auto_resolution"]["index"] == 1
    assert payload["output_device_request"] == "auto"
    assert payload["output_device_auto_resolution"]["index"] == 0
    assert [device["index"] for device in payload["devices"]] == [0, 1]
