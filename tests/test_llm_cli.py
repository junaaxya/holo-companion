import io
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from holo_companion.llm.base import CancellationHandle, ChatMessage, GenerationMetrics, GenerationResult, GenerationStatus, LlmConfig, LlmStreamEvent, TextChunk
from holo_companion.llm.cli import run_llm_cli


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeProvider:
    closed: bool = False

    async def stream(self, messages: tuple[ChatMessage, ...], cancellation: CancellationHandle | None = None) -> AsyncIterator[LlmStreamEvent]:
        yield TextChunk(text="Ha", index=0, elapsed_seconds=0.1)
        yield TextChunk(text="lo", index=1, elapsed_seconds=0.2)
        yield GenerationResult(text="Halo", status=GenerationStatus.COMPLETED, metrics=GenerationMetrics(http_headers_seconds=0.05, first_sse_event_seconds=0.08, time_to_first_token_seconds=0.1, total_seconds=0.3, chunk_count=2))

    async def aclose(self) -> None:
        self.closed = True


def test_llm_cli_streams_ndjson_chunks_before_result(tmp_path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = "https://api.example.test/v1"\nmodel = "model-a"\n', encoding="utf-8")
    stdout = io.StringIO()
    seen_configs: list[LlmConfig] = []

    provider = FakeProvider()

    def factory(config: LlmConfig) -> FakeProvider:
        seen_configs.append(config)
        return provider

    # When
    code = run_llm_cli(["--prompt", "Halo"], stdout, io.StringIO(), config_path=config_path, env={"LLM_API_KEY": "secret"}, provider_factory=factory)

    # Then
    assert code == 0
    assert seen_configs[0].model == "model-a"
    assert provider.closed is True
    assert [json.loads(line) for line in stdout.getvalue().splitlines()] == [
        {"type": "chunk", "index": 0, "text": "Ha", "elapsed_seconds": 0.1},
        {"type": "chunk", "index": 1, "text": "lo", "elapsed_seconds": 0.2},
        {"type": "result", "status": "completed", "text": "Halo", "metrics": {"http_headers_seconds": 0.05, "first_sse_event_seconds": 0.08, "time_to_first_token_seconds": 0.1, "total_seconds": 0.3, "chunk_count": 2}},
    ]


def test_llm_cli_reports_config_failure_without_traceback(tmp_path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = ""\nmodel = ""\n', encoding="utf-8")
    stderr = io.StringIO()

    # When
    code = run_llm_cli(["--prompt", "Halo"], io.StringIO(), stderr, config_path=config_path, env={})

    # Then
    assert code == 2
    assert "error: config:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_llm_cli_help_documents_prompt_only() -> None:
    # Given
    stdout = io.StringIO()

    # When
    code = run_llm_cli(["--help"], stdout, io.StringIO())

    # Then
    assert code == 0
    assert "llm-test --prompt TEXT" in stdout.getvalue()
    assert "--model" not in stdout.getvalue()
