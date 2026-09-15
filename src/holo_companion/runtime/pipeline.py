from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio

from holo_companion.llm.base import CancellationHandle
from holo_companion.runtime.playback import PlaybackAdmission, PlaybackSink
from holo_companion.tts.base import (
    StyleHints,
    SynthesisRequest,
    SynthesisResult,
    TTSAudioChunk,
    TTSProvider,
    TtsError,
    TtsProviderDiagnostic,
    format_empty_audio_diagnostic,
)

TEXT_QUEUE_CAPACITY = 2
AUDIO_QUEUE_CAPACITY = 2


@dataclass(frozen=True, slots=True)
class TextWorkItem:
    turn_id: int
    generation_id: int
    sequence: int
    text: str
    style: StyleHints | None


@dataclass(frozen=True, slots=True)
class AudioWorkItem:
    turn_id: int
    generation_id: int
    sequence: int
    chunk: TTSAudioChunk


TextSender = Callable[[TextWorkItem], Awaitable[None]]
TextProducer = Callable[[TextSender], Awaitable[None]]
CurrentGuard = Callable[[int, int], bool]
PipelineHook = Callable[[AudioWorkItem], Awaitable[None]]
TextPipelineHook = Callable[[TextWorkItem], Awaitable[None]]


@dataclass(slots=True)
class SpeechPipeline:
    tts: TTSProvider
    playback: PlaybackSink
    is_current: CurrentGuard
    on_audio_generated: PipelineHook | None = None
    after_playback: PipelineHook | None = None
    on_text_submitted: TextPipelineHook | None = None
    text_capacity: int = TEXT_QUEUE_CAPACITY
    audio_capacity: int = AUDIO_QUEUE_CAPACITY
    _played_items: list[AudioWorkItem] = field(default_factory=list, init=False)

    @property
    def played_items(self) -> tuple[AudioWorkItem, ...]:
        return tuple(self._played_items)

    async def run_text_to_audio(self, producer: TextProducer, cancellation: CancellationHandle) -> None:
        text_send, text_receive = anyio.create_memory_object_stream[TextWorkItem](self.text_capacity)
        audio_send, audio_receive = anyio.create_memory_object_stream[AudioWorkItem](self.audio_capacity)
        await self.playback.set_generation_active(True)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._produce_text, producer, text_send)
                tg.start_soon(self._synthesize_text, text_receive, audio_send, cancellation)
                tg.start_soon(self._play_audio, audio_receive)
        except BaseExceptionGroup as error:
            if unpacked := _unpack_task_group_error(error):
                raise unpacked
            raise
        finally:
            await self.playback.set_generation_active(False)

    async def _produce_text(self, producer: TextProducer, send: anyio.abc.ObjectSendStream[TextWorkItem]) -> None:
        async with send:
            await producer(send.send)

    async def _synthesize_text(
        self,
        receive: anyio.abc.ObjectReceiveStream[TextWorkItem],
        send: anyio.abc.ObjectSendStream[AudioWorkItem],
        cancellation: CancellationHandle,
    ) -> None:
        async with receive, send:
            async for item in receive:
                if not self.is_current(item.turn_id, item.generation_id):
                    continue
                request = SynthesisRequest(item.text, item.style)
                if self.on_text_submitted is not None:
                    await self.on_text_submitted(item)
                try:
                    await self._stream_tts(request, item, send, cancellation)
                except TtsError as error:
                    diag = error.diagnostic
                    if diag is not None and diag.error == "empty_audio":
                        cancelled_or_stale = cancellation.is_cancelled() or not self.is_current(item.turn_id, item.generation_id)
                        retry_triggered = not cancelled_or_stale
                        self._log_empty_audio_diagnostic(item, 0, diag, retry_triggered, cancelled_or_stale)
                        if retry_triggered:
                            try:
                                await self._stream_tts(request, item, send, cancellation)
                                continue
                            except TtsError as retry_error:
                                retry_diag = retry_error.diagnostic
                                retry_cancelled_or_stale = cancellation.is_cancelled() or not self.is_current(item.turn_id, item.generation_id)
                                self._log_empty_audio_diagnostic(item, 1, retry_diag, False, retry_cancelled_or_stale)
                                raise
                    raise

    def _log_empty_audio_diagnostic(
        self,
        item: TextWorkItem,
        retry_attempt: int,
        diagnostic: TtsProviderDiagnostic | None,
        retry_triggered: bool,
        cancelled_or_stale: bool,
    ) -> None:
        import sys
        print(
            format_empty_audio_diagnostic(
                turn_id=item.turn_id,
                segment_index=item.sequence,
                text_length=len(item.text),
                retry_attempt=retry_attempt,
                diagnostic=diagnostic,
                retry_triggered=retry_triggered,
                cancellation_or_staleness_active=cancelled_or_stale,
            ),
            file=sys.stderr,
            flush=True,
        )

    async def _stream_tts(
        self,
        request: SynthesisRequest,
        item: TextWorkItem,
        send: anyio.abc.ObjectSendStream[AudioWorkItem],
        cancellation: CancellationHandle,
    ) -> None:
        async for event in self.tts.stream(request, cancellation):
            if isinstance(event, TTSAudioChunk):
                if self.is_current(item.turn_id, item.generation_id):
                    audio_item = AudioWorkItem(item.turn_id, item.generation_id, item.sequence, event)
                    if self.on_audio_generated is not None:
                        await self.on_audio_generated(audio_item)
                    await send.send(audio_item)
            elif isinstance(event, SynthesisResult):
                pass

    async def _play_audio(self, receive: anyio.abc.ObjectReceiveStream[AudioWorkItem]) -> None:
        async with receive:
            async for item in receive:
                if not self.is_current(item.turn_id, item.generation_id):
                    continue
                await self.playback.write_and_start(PlaybackAdmission(item.turn_id, item.generation_id, item.chunk))
                if not self.is_current(item.turn_id, item.generation_id):
                    continue
                if self.after_playback is not None:
                    await self.after_playback(item)
                if self.is_current(item.turn_id, item.generation_id):
                    self._played_items.append(item)


def _unpack_task_group_error(error: BaseExceptionGroup) -> BaseException | None:
    leaves = _exception_leaves(error)
    if len(leaves) == 1:
        return leaves[0]
    tts_errors = [leaf for leaf in leaves if isinstance(leaf, TtsError)]
    if len(tts_errors) == 1 and all(isinstance(leaf, (TtsError, anyio.BrokenResourceError)) for leaf in leaves):
        return tts_errors[0]
    return None


def _exception_leaves(error: BaseExceptionGroup) -> tuple[BaseException, ...]:
    leaves: list[BaseException] = []
    for item in error.exceptions:
        if isinstance(item, BaseExceptionGroup):
            leaves.extend(_exception_leaves(item))
        else:
            leaves.append(item)
    return tuple(leaves)
