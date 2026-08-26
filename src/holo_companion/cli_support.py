import argparse
from pathlib import Path

from holo_companion.audio.types import AudioCliError
from holo_companion.stt.base import SttConfig, SttError


def parse_stt_config(args: argparse.Namespace) -> SttConfig:
    return SttConfig(
        model_name=args.model,
        cache_dir=args.model_cache,
        local_files_only=bool(args.local_files_only),
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )


def add_stt_flags(parser: argparse.ArgumentParser, *, default_language: str | None) -> None:
    parser.add_argument("--model", default="large-v3-turbo", choices=["large-v3-turbo", "large-v3", "small"])
    parser.add_argument("--language", default=default_language)
    parser.add_argument("--model-cache", default=Path("~/.cache/holo-companion/models"), type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-threads", default=0, type=parse_nonnegative_int)
    parser.add_argument("--num-workers", default=1, type=parse_positive_int)


def parse_nonnegative_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return value


def parse_positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def stt_error_to_audio_error(error: SttError) -> AudioCliError:
    return AudioCliError(str(error))
