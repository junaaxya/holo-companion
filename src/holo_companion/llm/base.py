from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit, urlunsplit

ProviderName: TypeAlias = Literal["openai_compatible"]
RoleName: TypeAlias = Literal["system", "user", "assistant"]
DEFAULT_MAX_CONTEXT_MESSAGES: Final = 6
DEFAULT_MAX_CONTEXT_CHARS: Final = 6_000
DEFAULT_MAX_PROMPT_CHARS: Final = 2_000
DEFAULT_TIMEOUT_SECONDS: Final = 30.0


@unique
class LlmErrorKind(StrEnum):
    CONFIG = "config"
    CANCELLED = "cancelled"
    NETWORK = "network"
    HTTP_STATUS = "http_status"
    PROTOCOL = "protocol"


@unique
class GenerationStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class LlmError(Exception):
    __slots__ = ("_frozen", "kind", "message", "status_code")

    def __init__(self, kind: LlmErrorKind, message: str, status_code: int | None = None) -> None:
        super().__init__(kind, message, status_code)
        super().__setattr__("kind", kind)
        super().__setattr__("message", message)
        super().__setattr__("status_code", status_code)
        super().__setattr__("_frozen", True)

    def __setattr__(self, name: str, value: LlmErrorKind | str | int | bool | None) -> None:
        if name in {"__traceback__", "__cause__", "__context__", "__suppress_context__"}:
            super().__setattr__(name, value)
            return
        if getattr(self, "_frozen", False):
            raise AttributeError("LlmError is immutable")
        super().__setattr__(name, value)

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.message}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: RoleName
    content: str

    def __post_init__(self) -> None:
        if self.content.strip() == "":
            raise LlmError(LlmErrorKind.CONFIG, "chat message content must be nonblank")


@dataclass(frozen=True, slots=True)
class LlmConfig:
    provider: ProviderName
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.provider != "openai_compatible":
            raise LlmError(LlmErrorKind.CONFIG, f"unsupported LLM provider: {self.provider}")
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        if self.model.strip() == "":
            raise LlmError(LlmErrorKind.CONFIG, "LLM model must be nonblank")
        object.__setattr__(self, "model", self.model.strip())
        if self.api_key.strip() == "":
            raise LlmError(LlmErrorKind.CONFIG, "LLM_API_KEY must be set")
        object.__setattr__(self, "api_key", self.api_key.strip())
        if self.timeout_seconds <= 0:
            raise LlmError(LlmErrorKind.CONFIG, "LLM timeout_seconds must be greater than 0")


class CancellationHandle:
    __slots__ = ("_cancelled", "_reason")

    def __init__(self) -> None:
        self._cancelled = False
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancelled = True
        self._reason = reason

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise LlmError(LlmErrorKind.CANCELLED, self._reason)


CancellationToken = CancellationHandle


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    index: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    http_headers_seconds: float | None
    first_sse_event_seconds: float | None
    time_to_first_token_seconds: float | None
    total_seconds: float
    chunk_count: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    status: GenerationStatus
    metrics: GenerationMetrics


LlmStreamEvent: TypeAlias = TextChunk | GenerationResult


class LLMProvider(Protocol):
    def stream(self, messages: tuple[ChatMessage, ...], cancellation: CancellationHandle | None = None) -> AsyncIterator[LlmStreamEvent]: ...

    async def aclose(self) -> None: ...


def normalize_base_url(raw_url: str) -> str:
    normalized = raw_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise LlmError(LlmErrorKind.CONFIG, "LLM base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise LlmError(LlmErrorKind.CONFIG, "LLM base_url must not contain credentials")
    if parsed.query != "" or parsed.fragment != "":
        raise LlmError(LlmErrorKind.CONFIG, "LLM base_url must not contain query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
