import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest

from holo_companion.llm.base import CancellationHandle, ChatMessage, GenerationResult, GenerationStatus, LlmConfig, LlmError, LlmErrorKind, TextChunk
from holo_companion.llm.openai_compatible import OpenAiCompatibleProvider


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeClock:
    values: list[float]

    def __call__(self) -> float:
        return self.values.pop(0)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def provider_with_response(response: httpx.Response, clock: FakeClock) -> tuple[OpenAiCompatibleProvider, list[httpx.Request]]:
    # Given
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response.request = request
        return response

    config = LlmConfig(provider="openai_compatible", base_url="https://api.example.test/v1", model="model-a", api_key="secret")
    client = httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler))
    return OpenAiCompatibleProvider(config=config, client=client, monotonic_seconds=clock), requests


@pytest.mark.anyio
async def test_openai_provider_streams_fragments_multiline_comments_and_metrics() -> None:
    # Given
    body = [
        b": comment\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}",
        b"\n\n",
        b"data: {\"choices\":[{\"delta\":{}}]}\n\n",
        b"data: [DONE]\n\n",
    ]
    response = httpx.Response(200, stream=ChunkStream(body))
    provider, requests = provider_with_response(response, FakeClock([10.0, 10.2, 10.5, 10.7, 11.0, 11.2]))

    # When
    events = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]

    # Then
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://api.example.test/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content) == {"model": "model-a", "messages": [{"role": "user", "content": "Halo"}], "stream": True}
    assert events == [
        TextChunk(text="Ha", index=0, elapsed_seconds=0.6999999999999993),
        TextChunk(text="lo", index=1, elapsed_seconds=1.0),
        GenerationResult(text="Halo", status=GenerationStatus.COMPLETED, metrics=events[-1].metrics),
    ]
    result = events[-1]
    assert isinstance(result, GenerationResult)
    assert result.metrics.http_headers_seconds == 0.1999999999999993
    assert result.metrics.first_sse_event_seconds == 0.5
    assert result.metrics.time_to_first_token_seconds == 0.6999999999999993
    assert result.metrics.total_seconds == 1.1999999999999993
    assert result.metrics.chunk_count == 2


@pytest.mark.anyio
async def test_openai_provider_cancels_before_request() -> None:
    # Given
    response = httpx.Response(200, stream=ChunkStream([]))
    provider, requests = provider_with_response(response, FakeClock([1.0]))
    cancellation = CancellationHandle()
    cancellation.cancel("barge-in")

    # When / Then
    with pytest.raises(LlmError) as raised:
        _ = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),), cancellation)]
    assert raised.value.kind is LlmErrorKind.CANCELLED
    assert requests == []


@pytest.mark.anyio
async def test_openai_provider_returns_cancelled_result_when_cancelled_after_chunk() -> None:
    # Given
    stream = ChunkStream([
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
    ])
    provider, _requests = provider_with_response(httpx.Response(200, stream=stream), FakeClock([10.0, 10.2, 10.4, 10.6, 10.8]))
    cancellation = CancellationHandle()

    # When
    events = provider.stream((ChatMessage(role="user", content="Halo"),), cancellation)
    first = await anext(events)
    cancellation.cancel("barge-in")
    result = await anext(events)

    # Then
    assert first == TextChunk(text="Ha", index=0, elapsed_seconds=0.5999999999999996)
    assert result == GenerationResult(text="Ha", status=GenerationStatus.CANCELLED, metrics=result.metrics)
    assert result.metrics.chunk_count == 1
    assert stream.closed


@pytest.mark.anyio
async def test_openai_provider_maps_http_status() -> None:
    # Given
    provider, _requests = provider_with_response(httpx.Response(429, text="rate limited"), FakeClock([1.0, 1.1]))

    # When / Then
    with pytest.raises(LlmError) as raised:
        _ = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]
    assert raised.value.kind is LlmErrorKind.HTTP_STATUS
    assert raised.value.status_code == 429


@pytest.mark.anyio
async def test_openai_provider_maps_invalid_json_protocol_error() -> None:
    # Given
    provider, _requests = provider_with_response(httpx.Response(200, stream=ChunkStream([b"data: nope\n\n"])), FakeClock([1.0, 1.1, 1.2]))

    # When / Then
    with pytest.raises(LlmError) as raised:
        _ = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]
    assert raised.value.kind is LlmErrorKind.PROTOCOL


@pytest.mark.anyio
async def test_openai_provider_maps_eof_without_done_or_finish_reason_to_protocol_error() -> None:
    # Given
    provider, _requests = provider_with_response(
        httpx.Response(200, stream=ChunkStream([b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"}}]}\n\n"])),
        FakeClock([1.0, 1.1, 1.2, 1.3]),
    )

    # When / Then
    with pytest.raises(LlmError) as raised:
        _ = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]
    assert raised.value.kind is LlmErrorKind.PROTOCOL


@pytest.mark.anyio
async def test_openai_provider_treats_choice_finish_reason_as_completed_without_done() -> None:
    # Given
    body = [
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"},\"finish_reason\":\"stop\"}]}\n\n",
    ]
    provider, _requests = provider_with_response(httpx.Response(200, stream=ChunkStream(body)), FakeClock([10.0, 10.2, 10.4, 10.6, 10.8]))

    # When
    events = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]

    # Then
    assert events == [
        TextChunk(text="Ha", index=0, elapsed_seconds=0.5999999999999996),
        GenerationResult(text="Ha", status=GenerationStatus.COMPLETED, metrics=events[-1].metrics),
    ]


@pytest.mark.anyio
async def test_openai_provider_measures_headers_first_raw_sse_and_first_visible_delta() -> None:
    # Given
    body = [
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"}}]}\n\n",
        b"data: [DONE]\n\n",
    ]
    provider, _requests = provider_with_response(httpx.Response(200, stream=ChunkStream(body)), FakeClock([10.0, 10.1, 10.2, 10.4, 10.6]))

    # When
    events = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]

    # Then
    assert events[0] == TextChunk(text="Ha", index=0, elapsed_seconds=0.40000000000000036)
    result = events[-1]
    assert isinstance(result, GenerationResult)
    assert result.metrics.http_headers_seconds == 0.09999999999999964
    assert result.metrics.first_sse_event_seconds == 0.1999999999999993
    assert result.metrics.time_to_first_token_seconds == 0.40000000000000036
    assert result.metrics.total_seconds == 0.5999999999999996
    assert result.metrics.chunk_count == 1


@pytest.mark.anyio
async def test_openai_provider_forwards_first_visible_delta_before_stream_completion() -> None:
    # Given
    body = [
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Ha\"}}]}\n\n",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        b"data: [DONE]\n\n",
    ]
    provider, _requests = provider_with_response(httpx.Response(200, stream=ChunkStream(body)), FakeClock([10.0, 10.1, 10.2, 10.3, 10.4]))

    # When
    events = provider.stream((ChatMessage(role="user", content="Halo"),))
    first = await anext(events)
    second = await anext(events)

    # Then
    assert first == TextChunk(text="Ha", index=0, elapsed_seconds=0.3000000000000007)
    assert second == TextChunk(text="lo", index=1, elapsed_seconds=0.40000000000000036)


@pytest.mark.anyio
async def test_openai_provider_maps_transport_error() -> None:
    # Given
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    config = LlmConfig(provider="openai_compatible", base_url="https://api.example.test/v1", model="model-a", api_key="secret")
    provider = OpenAiCompatibleProvider(config=config, client=httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler)), monotonic_seconds=FakeClock([1.0]))

    # When / Then
    with pytest.raises(LlmError) as raised:
        _ = [event async for event in provider.stream((ChatMessage(role="user", content="Halo"),))]
    assert raised.value.kind is LlmErrorKind.NETWORK
