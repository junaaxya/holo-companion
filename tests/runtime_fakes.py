from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anyio

from holo_companion.llm.base import ChatMessage, GenerationMetrics, GenerationResult, GenerationStatus, LlmError, LlmErrorKind, LlmStreamEvent, TextChunk
from holo_companion.runtime.playback import PlaybackAdmission, PlaybackSink
from holo_companion.stt.base import SttError, Transcript, Utterance
from holo_companion.tts.base import SynthesisMetrics, SynthesisRequest, SynthesisResult, SynthesisStatus, TTSAudioChunk, TtsError, TtsErrorKind, TtsStreamEvent


@dataclass(slots=True)
class FakeStt:
    result: Transcript
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    active: int = 0
    max_active: int = 0
    closed: bool = False
    generation_active: bool = False
    fail: bool = False
    bug: bool = False

    def preload(self) -> float:
        return 0.0

    def transcribe(self, value: Utterance, language_hint: str | None = None) -> Transcript:
        del value, language_hint
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            if self.release is not None:
                anyio.from_thread.run(self.release.wait)
            if self.bug:
                raise ValueError("stt programming bug")
            if self.fail:
                raise SttError("stt failed")
            return self.result
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True

    async def set_generation_active(self, active: bool) -> None:
        self.generation_active = active


@dataclass(slots=True)
class FakeLlm:
    chunks: tuple[str, ...] = ("hai",)
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    cancelled: bool = False
    closed: bool = False
    fail: bool = False
    bug: bool = False

    async def stream(self, messages: tuple[ChatMessage, ...], cancellation: object | None = None) -> AsyncIterator[LlmStreamEvent]:
        del messages
        self.entered.set()
        if self.release is not None:
            try:
                await self.release.wait()
            finally:
                self.cancelled = self.cancelled or bool(getattr(cancellation, "is_cancelled", lambda: False)())
        if self.bug:
            raise ValueError("llm programming bug")
        if self.fail:
            raise LlmError(LlmErrorKind.PROTOCOL, "llm failed")
        for index, text in enumerate(self.chunks):
            yield TextChunk(text=text, index=index, elapsed_seconds=0.01)
        yield GenerationResult("".join(self.chunks), GenerationStatus.COMPLETED, GenerationMetrics(0.01, 0.01, 0.01, 0.02, len(self.chunks)))

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakeTts:
    chunk: TTSAudioChunk
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    requests: list[SynthesisRequest] = field(default_factory=list)
    cancelled: bool = False
    closed: bool = False
    fail: bool = False
    bug: bool = False

    async def stream(self, request: SynthesisRequest, cancellation: object | None = None) -> AsyncIterator[TtsStreamEvent]:
        self.entered.set()
        self.requests.append(request)
        if self.release is not None:
            try:
                await self.release.wait()
            finally:
                self.cancelled = self.cancelled or bool(getattr(cancellation, "is_cancelled", lambda: False)())
        if self.bug:
            raise ValueError("tts programming bug")
        if self.fail:
            raise TtsError(TtsErrorKind.PROVIDER, "tts failed")
        yield self.chunk
        yield SynthesisResult(SynthesisRequest("hai"), SynthesisStatus.COMPLETED, SynthesisMetrics(0.0, 0.01, 0.02, 0.001, 20.0, 1))

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakePlayback(PlaybackSink):
    started: anyio.Event = field(default_factory=anyio.Event)
    stopped: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    admissions: list[PlaybackAdmission] = field(default_factory=list)
    active: int = 0
    max_active: int = 0
    closed: bool = False
    on_samples_consumed: object | None = None

    async def write_and_start(self, admission: PlaybackAdmission) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.admissions.append(admission)
        self.started.set()
        if callable(self.on_samples_consumed):
            self.on_samples_consumed(admission.chunk.samples, admission.chunk.audio_format.sample_rate_hz)
        try:
            if self.release is not None:
                await self.release.wait()
        finally:
            self.active -= 1

    async def stop(self) -> None:
        self.stopped.set()
        if self.release is not None:
            self.release.set()

    async def aclose(self) -> None:
        self.closed = True

    async def wait_until_drained(self) -> None:
        pass
