import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from enum import StrEnum, unique
from pathlib import Path
from typing import Final, TextIO, assert_never

from holo_companion.audio.capture import (
    AudioBackend,
    SoundDeviceBackend,
    SoundFileWaveWriter,
    WaveWriter,
    enumerate_devices,
    record_wav,
)
from holo_companion.audio.types import AudioCliError, CaptureRequest, DeviceInfo, Seconds

PROGRAM_NAME: Final = "holo_companion"
DEFAULT_RECORD_SECONDS: Final = 5.0
MIN_RECORD_SECONDS: Final = 0.1


@unique
class Command(StrEnum):
    DEVICES = "devices"
    RECORD = "record"


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv, backend=SoundDeviceBackend(), writer=SoundFileWaveWriter(), stdout=sys.stdout, stderr=sys.stderr)


def run_cli(
    argv: Sequence[str] | None,
    backend: AudioBackend,
    writer: WaveWriter,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        command = parse_command(args.command)
        match command:
            case Command.DEVICES:
                payload = devices_payload(backend)
            case Command.RECORD:
                payload = record_payload(args, backend, writer)
            case None:
                parser.print_help(stdout)
                return 0
            case unreachable:
                assert_never(unreachable)
    except AudioCliError as error:
        print(f"error: {error}", file=stderr)
        return 2
    json.dump(payload, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("devices", help="List current PortAudio devices and defaults")
    record_parser = subcommands.add_parser("record", help="Record a short mono 16 kHz WAV")
    record_parser.add_argument("--seconds", default=DEFAULT_RECORD_SECONDS, type=parse_seconds)
    record_parser.add_argument("--output", required=True, type=Path)
    record_parser.add_argument("--input-device", default="auto")
    record_parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_command(command: str | None) -> Command | None:
    if command is None:
        return None
    try:
        return Command(command)
    except ValueError as error:
        raise AudioCliError(f"unknown command: {command}") from error


def devices_payload(backend: AudioBackend) -> dict[str, str | list[dict[str, int | float | str]] | dict[str, int | float | str]]:
    devices, defaults = enumerate_devices(backend)
    return {
        "input_device_request": "auto",
        "input_device_auto_resolution": device_payload(defaults.input_selection.resolved),
        "output_device_request": "auto",
        "output_device_auto_resolution": device_payload(defaults.output_selection.resolved),
        "devices": [device_payload(device) for device in devices],
    }


def record_payload(
    args: argparse.Namespace,
    backend: AudioBackend,
    writer: WaveWriter,
) -> dict[str, str | dict[str, int | float | str | bool]]:
    input_device = parse_input_device(args.input_device)
    output_path = parse_output_path(args.output)
    request = CaptureRequest(
        seconds=args.seconds,
        output_path=output_path,
        requested_input_device=input_device,
        overwrite=bool(args.overwrite),
    )
    result = record_wav(request, backend, writer)
    return {
        "requested_input_device": result.requested_input_device,
        "resolved_input_device": device_payload(result.resolved_input_device),
        "format": asdict(result.format),
        "output_path": str(result.output_path),
        "diagnostics": asdict(result.diagnostics),
    }


def parse_seconds(raw_seconds: str) -> Seconds:
    try:
        seconds = float(raw_seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seconds must be a number") from error
    if seconds < MIN_RECORD_SECONDS:
        raise argparse.ArgumentTypeError("--seconds must be at least 0.1")
    return Seconds(seconds)


def parse_input_device(input_device: str) -> str:
    normalized = input_device.strip()
    if normalized == "":
        raise AudioCliError("--input-device must be 'auto' or a device-name query")
    if normalized.isdecimal():
        raise AudioCliError("--input-device accepts a device-name query, not a numeric device id")
    return normalized


def parse_output_path(path: Path) -> Path:
    if path.exists() and path.is_dir():
        raise AudioCliError(f"--output must be a WAV file path, not a directory: {path}")
    if path.suffix.lower() != ".wav":
        raise AudioCliError("--output must end with .wav")
    return path


def device_payload(device: DeviceInfo) -> dict[str, int | float | str]:
    return {
        "index": int(device.index),
        "name": device.name,
        "max_input_channels": device.max_input_channels,
        "max_output_channels": device.max_output_channels,
        "default_samplerate": device.default_samplerate,
    }
