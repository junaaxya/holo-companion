import json
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

import httpx

from holo_companion.llm.base import CancellationHandle, ChatMessage, GenerationMetrics, GenerationResult, GenerationStatus, LlmConfig, LlmError, LlmErrorKind, LlmStreamEvent, TextChunk
from holo_companion.llm.payloads import message_payload

CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
FinishReason: TypeAlias = str | int | float | bool | list["FinishReason"] | dict[str, "FinishReason"]


class Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    text: str
    finished: bool


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleProvider:
    config: LlmConfig
    client: httpx.AsyncClient
    monotonic_seconds: Clock

    async def stream(self, messages: tuple[ChatMessage, ...], cancellation: CancellationHandle | None = None) -> AsyncIterator[LlmStreamEvent]:
        handle = CancellationHandle() if cancellation is None else cancellation
        handle.raise_if_cancelled()
        start = self.monotonic_seconds()
        chunks: list[str] = []
        http_headers_seconds: float | None = None
        first_sse_event_seconds: float | None = None
        first_token_seconds: float | None = None
        index = 0
        completed = False
        try:
            async with self.client.stream("POST", CHAT_COMPLETIONS_PATH, json={"model": self.config.model, "messages": [message_payload(message) for message in messages], "stream": True}, headers={"Authorization": f"Bearer {self.config.api_key}"}) as response:
                http_headers_seconds = self.monotonic_seconds() - start
                if response.status_code >= 400:
                    raise LlmError(LlmErrorKind.HTTP_STATUS, f"LLM provider returned HTTP {response.status_code}", status_code=response.status_code)
                if not handle.is_cancelled():
                    async for data in sse_data_events(response):
                        if first_sse_event_seconds is None:
                            first_sse_event_seconds = self.monotonic_seconds() - start
                        if handle.is_cancelled():
                            break
                        if data == "[DONE]":
                            completed = True
                            break
                        delta = parse_delta_event(data)
                        completed = delta.finished
                        text = delta.text
                        if text == "":
                            if completed:
                                break
                            continue
                        now = self.monotonic_seconds()
                        if first_token_seconds is None:
                            first_token_seconds = now - start
                        chunks.append(text)
                        yield TextChunk(text=text, index=index, elapsed_seconds=now - start)
                        index += 1
                        if handle.is_cancelled() or completed:
                            break
        except httpx.HTTPError as error:
            raise LlmError(LlmErrorKind.NETWORK, str(error)) from error
        if not completed and not handle.is_cancelled():
            raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream ended before completion marker")
        total_seconds = self.monotonic_seconds() - start
        status = GenerationStatus.CANCELLED if handle.is_cancelled() else GenerationStatus.COMPLETED
        yield GenerationResult(text="".join(chunks), status=status, metrics=GenerationMetrics(http_headers_seconds=http_headers_seconds, first_sse_event_seconds=first_sse_event_seconds, time_to_first_token_seconds=first_token_seconds, total_seconds=total_seconds, chunk_count=index))

    async def aclose(self) -> None:
        await self.client.aclose()


def create_httpx_client(config: LlmConfig) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30.0)
    timeout = httpx.Timeout(connect=min(10.0, config.timeout_seconds), read=None, write=10.0, pool=10.0)
    transport = httpx.AsyncHTTPTransport(http2=True, retries=3, limits=limits, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)])
    return httpx.AsyncClient(base_url=config.base_url, transport=transport, timeout=timeout, follow_redirects=True)


def provider_from_config(config: LlmConfig, monotonic_seconds: Clock) -> OpenAiCompatibleProvider:
    return OpenAiCompatibleProvider(config=config, client=create_httpx_client(config), monotonic_seconds=monotonic_seconds)


async def sse_data_events(response: httpx.Response) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        yield "\n".join(data_lines)


def parse_delta_event(raw_data: str) -> DeltaEvent:
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as error:
        raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream emitted invalid JSON") from error
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream payload missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream choice must be an object")
    delta = first_choice.get("delta", {})
    if not isinstance(delta, dict):
        raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream delta must be an object")
    content = delta.get("content", "")
    if not isinstance(content, str):
        raise LlmError(LlmErrorKind.PROTOCOL, "LLM stream content must be text")
    finish_reason: FinishReason | None = first_choice.get("finish_reason")
    return DeltaEvent(text=content, finished=finish_reason is not None)
