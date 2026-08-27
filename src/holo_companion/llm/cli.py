import json
import os
import sys
import time
from collections.abc import Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Protocol, TextIO, assert_never

import anyio

from holo_companion.llm.base import CancellationHandle, GenerationResult, LLMProvider, LlmConfig, LlmError, TextChunk
from holo_companion.llm.config import CONFIG_PATH, load_llm_config
from holo_companion.llm.context import ConversationContext
from holo_companion.llm.openai_compatible import provider_from_config
from holo_companion.llm.payloads import JsonObject, chunk_payload, result_payload


class LlmProviderFactory(Protocol):
    def __call__(self, config: LlmConfig) -> LLMProvider: ...


def main(argv: Sequence[str] | None = None) -> int:
    return run_llm_cli(argv, stdout=sys.stdout, stderr=sys.stderr)


def run_llm_cli(
    argv: Sequence[str] | None,
    stdout: TextIO,
    stderr: TextIO,
    *,
    config_path: Path = CONFIG_PATH,
    env: dict[str, str] | None = None,
    provider_factory: LlmProviderFactory | None = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--help"] or args == ["-h"]:
        print_help(stdout)
        return 0
    if len(args) != 2 or args[0] != "--prompt":
        print("error: expected --prompt TEXT", file=stderr)
        return 2
    try:
        config = load_llm_config(config_path, os.environ if env is None else env)
        factory = provider_factory or (lambda loaded_config: provider_from_config(loaded_config, time.monotonic))
        return anyio.run(_run_stream, args[1], config, factory, stdout)
    except LlmError as error:
        print(f"error: {error}", file=stderr)
        return 2


async def _run_stream(prompt: str, config: LlmConfig, provider_factory: LlmProviderFactory, stdout: TextIO) -> int:
    provider = provider_factory(config)
    context = ConversationContext().with_user_prompt(prompt)
    try:
        async with aclosing(provider.stream(context.provider_messages(), CancellationHandle())) as stream:
            async for event in stream:
                match event:
                    case TextChunk() as chunk:
                        write_ndjson(stdout, chunk_payload(chunk))
                    case GenerationResult() as result:
                        write_ndjson(stdout, result_payload(result))
                    case unreachable:
                        assert_never(unreachable)
    finally:
        await provider.aclose()
    return 0


def write_ndjson(stdout: TextIO, payload: JsonObject) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    stdout.write("\n")
    stdout.flush()


def print_help(stdout: TextIO) -> None:
    stdout.write("usage: llm-test --prompt TEXT\n\nStream OpenAI-compatible LLM output as NDJSON. Reads provider config from config/default.toml and LLM_API_KEY from the environment.\n")
