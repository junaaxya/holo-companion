import argparse
import json
import sys
from collections.abc import Sequence
from enum import StrEnum, unique
from pathlib import Path
from typing import Final, TextIO, assert_never

from holo_companion.audio.capture import (
    AudioBackend,
    SoundDeviceBackend,
    SoundFileWaveWriter,
    WaveWriter,
)
from holo_companion.audio.types import AudioCliError, Seconds
from holo_companion.cli_commands import devices_payload, record_payload, stt_benchmark_payload, stt_wav_payload, vad_test_payload
from holo_companion.cli_support import add_stt_flags
from holo_companion.stt.base import SttError
from holo_companion.stt.payloads import provider_from_config
from holo_companion.vad.diagnostics import VadProviderFactory
from holo_companion.vad.silero import create_silero_vad_provider

PROGRAM_NAME: Final = "holo_companion"
DEFAULT_RECORD_SECONDS: Final = 5.0
MIN_RECORD_SECONDS: Final = 0.1
DEFAULT_VAD_SECONDS: Final = 10.0
DEFAULT_VAD_THRESHOLD: Final = 0.6
DEFAULT_MIN_SPEECH_MS: Final = 500
DEFAULT_MIN_SILENCE_MS: Final = 700
DEFAULT_SPEECH_PAD_MS: Final = 30

@unique
class Command(StrEnum):
    DEVICES = "devices"
    RECORD = "record"
    VAD_TEST = "vad-test"
    STT_WAV = "stt-wav"
    STT_BENCHMARK = "stt-benchmark"


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        argv,
        backend=SoundDeviceBackend(),
        writer=SoundFileWaveWriter(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        vad_provider_factory=create_silero_vad_provider,
        stt_provider_factory=provider_from_config,
    )


def run_cli(
    argv: Sequence[str] | None,
    backend: AudioBackend,
    writer: WaveWriter,
    stdout: TextIO,
    stderr: TextIO,
    vad_provider_factory: VadProviderFactory = create_silero_vad_provider,
    stt_provider_factory=provider_from_config,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        command = parse_command(args.command)
        match command:
            case Command.DEVICES:
                payload = devices_payload(backend)
            case Command.RECORD:
                payload = record_payload(args, backend, writer)
            case Command.VAD_TEST:
                payload = vad_test_payload(args, backend, vad_provider_factory, stt_provider_factory)
            case Command.STT_WAV:
                payload = stt_wav_payload(args, stt_provider_factory)
            case Command.STT_BENCHMARK:
                payload = stt_benchmark_payload(args, stt_provider_factory)
            case None:
                parser.print_help(stdout)
                return 0
            case unreachable:
                assert_never(unreachable)
    except (AudioCliError, SttError) as error:
        print(f"error: {error}", file=stderr)
        return 2
    json.dump(payload, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(prog=PROGRAM_NAME)
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("devices", help="List current PortAudio devices and defaults")
    record_parser = subcommands.add_parser("record", help="Record a short mono 16 kHz WAV")
    record_parser.add_argument("--seconds", default=DEFAULT_RECORD_SECONDS, type=parse_seconds)
    record_parser.add_argument("--output", required=True, type=Path)
    record_parser.add_argument("--input-device", default="auto")
    record_parser.add_argument("--overwrite", action="store_true")
    vad_parser = subcommands.add_parser("vad-test", help="Run finite streaming Silero VAD diagnostics")
    vad_parser.add_argument("--seconds", default=DEFAULT_VAD_SECONDS, type=parse_seconds)
    vad_parser.add_argument("--input-device", default="auto")
    vad_parser.add_argument("--threshold", default=DEFAULT_VAD_THRESHOLD, type=parse_threshold)
    vad_parser.add_argument("--min-speech-ms", default=DEFAULT_MIN_SPEECH_MS, type=parse_positive_milliseconds)
    vad_parser.add_argument("--min-silence-ms", default=DEFAULT_MIN_SILENCE_MS, type=parse_positive_milliseconds)
    vad_parser.add_argument("--speech-pad-ms", default=DEFAULT_SPEECH_PAD_MS, type=parse_nonnegative_milliseconds)
    vad_parser.add_argument("--transcribe", action="store_true")
    add_stt_flags(vad_parser, default_language="id")
    stt_wav_parser = subcommands.add_parser("stt-wav", help="Transcribe one or more canonical mono 16 kHz WAV files")
    stt_wav_parser.add_argument("--input", action="append", required=True, type=Path)
    add_stt_flags(stt_wav_parser, default_language="id")
    benchmark_parser = subcommands.add_parser("stt-benchmark", help="Run a manifest-based STT benchmark")
    benchmark_parser.add_argument("--manifest", required=True, type=Path)
    add_stt_flags(benchmark_parser, default_language=None)
    return parser


def parse_command(command: str | None) -> Command | None:
    if command is None:
        return None
    try:
        return Command(command)
    except ValueError as error:
        raise AudioCliError(f"unknown command: {command}") from error


def parse_seconds(raw_seconds: str) -> Seconds:
    try:
        seconds = float(raw_seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seconds must be a number") from error
    if seconds < MIN_RECORD_SECONDS:
        raise argparse.ArgumentTypeError("--seconds must be at least 0.1")
    return Seconds(seconds)


def parse_threshold(raw_threshold: str) -> float:
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--threshold must be a number") from error
    if threshold <= 0 or threshold >= 1:
        raise argparse.ArgumentTypeError("--threshold must be greater than 0 and less than 1")
    return threshold


def parse_positive_milliseconds(raw_milliseconds: str) -> int:
    try:
        milliseconds = int(raw_milliseconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError("milliseconds must be an integer") from error
    if milliseconds <= 0:
        raise argparse.ArgumentTypeError("milliseconds must be greater than 0")
    return milliseconds


def parse_nonnegative_milliseconds(raw_milliseconds: str) -> int:
    try:
        milliseconds = int(raw_milliseconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError("milliseconds must be an integer") from error
    if milliseconds < 0:
        raise argparse.ArgumentTypeError("milliseconds must be at least 0")
    return milliseconds


class CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AudioCliError(message)
