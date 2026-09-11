from collections.abc import AsyncIterator
from typing import Protocol

import httpx


class HttpStreamContext(Protocol):
    async def __aenter__(self) -> httpx.Response: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...


class ElevenLabsHttpConnector(Protocol):
    def stream_text_to_dialogue(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> HttpStreamContext: ...

    async def aclose(self) -> None: ...


class DefaultHttpConnector:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30.0)
        timeout = httpx.Timeout(connect=min(10.0, timeout_seconds), read=None, write=10.0, pool=10.0)
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)

    def stream_text_to_dialogue(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> HttpStreamContext:
        return self._client.stream("POST", url, headers=headers, json=payload)

    async def aclose(self) -> None:
        await self._client.aclose()
