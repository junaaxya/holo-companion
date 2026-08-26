import argparse
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

from holo_companion.audio.capture import AudioBackend, WaveWriter, enumerate_devices, record_wav
from holo_companion.audio.types import AudioCliError, CaptureRequest, DeviceInfo
from holo_companion.cli_support import parse_stt_config
from holo_companion.stt.benchmark import benchmark_payload as build_benchmark_payload
from holo_companion.stt.payloads import JsonObject, SttProviderFactory, provider_from_config, stt_wav_payload as build_stt_wav_payload
from holo_companion.vad.diagnostics import VadProviderFactory, vad_test_payload as build_vad_test_payload

DeviceJson: TypeAlias = dict[str, int | float | str]


def devices_payload(backend: AudioBackend) -> JsonObject:
    devices, defaults = enumerate_devices(backend)
    return {
        "input_device_request": "auto",
        "input_device_auto_resolution": device_payload(defaults.input_selection.resolved),
        "output_device_request": "auto",
        "output_device_auto_resolution": device_payload(defaults.output_selection.resolved),
        "devices": [device_payload(device) for device in devices],
    }


def record_payload(args: argparse.Namespace, backend: AudioBackend, writer: WaveWriter) -> JsonObject:
    request = CaptureRequest(seconds=args.seconds, output_path=parse_output_path(args.output), requested_input_device=parse_input_device(args.input_device), overwrite=bool(args.overwrite))
    result = record_wav(request, backend, writer)
    return {
        "requested_input_device": result.requested_input_device,
        "resolved_input_device": device_payload(result.resolved_input_device),
        "format": asdict(result.format),
        "output_path": str(result.output_path),
        "diagnostics": asdict(result.diagnostics),
    }


def vad_test_payload(args: argparse.Namespace, backend: AudioBackend, vad_provider_factory: VadProviderFactory, stt_provider_factory: SttProviderFactory = provider_from_config) -> JsonObject:
    return build_vad_test_payload(args, backend, vad_provider_factory, stt_provider_factory=stt_provider_factory)


def stt_wav_payload(args: argparse.Namespace, stt_provider_factory: SttProviderFactory = provider_from_config) -> JsonObject:
    return build_stt_wav_payload(args.input, parse_stt_config(args), args.language, stt_provider_factory)


def stt_benchmark_payload(args: argparse.Namespace, stt_provider_factory: SttProviderFactory = provider_from_config) -> JsonObject:
    return build_benchmark_payload(args.manifest, parse_stt_config(args), stt_provider_factory)


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


def device_payload(device: DeviceInfo) -> DeviceJson:
    return {
        "index": int(device.index),
        "name": device.name,
        "max_input_channels": device.max_input_channels,
        "max_output_channels": device.max_output_channels,
        "default_samplerate": device.default_samplerate,
    }
