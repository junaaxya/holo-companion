from collections.abc import Callable
from dataclasses import dataclass

from holo_companion.llm.base import CancellationHandle, ChatMessage, GenerationResult, LLMProvider, TextChunk
from holo_companion.runtime.metrics import MonotonicClock, TurnTimestamps
from holo_companion.runtime.ownership import OwnedTurn, TurnCancelledError
from holo_companion.runtime.pipeline import AudioWorkItem, SpeechPipeline, TextSender, TextWorkItem
from holo_companion.runtime.playback import PlaybackSink
from holo_companion.runtime.segmenter import IncrementalSpeechSegmenter
from holo_companion.runtime.spoken_text import normalize_spoken_text
from holo_companion.runtime.state import RuntimeState
from holo_companion.runtime.turn import TextSegmentDiagnostic, TextTurnDiagnostic
from holo_companion.runtime.timestamps import with_first_llm_raw_delta, with_first_llm_token, with_first_playback, with_first_tts_audio, with_tts_request
from holo_companion.tts.base import StyleHints, TTSProvider


@dataclass(slots=True)
class SpeakContext:
    llm: LLMProvider
    tts: TTSProvider
    playback: PlaybackSink
    clock: MonotonicClock
    style: StyleHints
    move: Callable[[RuntimeState], None]
    ensure_current: Callable[[OwnedTurn], None]
    is_current: Callable[[int, int], bool]
    get_state: Callable[[], RuntimeState]


async def generate_and_speak(
    ctx: SpeakContext,
    owned: OwnedTurn,
    messages: tuple[ChatMessage, ...],
    assistant_parts: list[str],
    timestamps: TurnTimestamps,
) -> tuple[TurnTimestamps, TextTurnDiagnostic]:
    ctx.move(RuntimeState.THINKING)
    updated = TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=ctx.clock.now(),
    )

    async def on_audio_generated(item: AudioWorkItem) -> None:
        del item
        nonlocal updated
        if updated.first_tts_audio is None:
            updated = with_first_tts_audio(updated, ctx.clock.now())

    async def after_playback(item: AudioWorkItem) -> None:
        del item
        nonlocal updated
        if updated.first_playback_audio is None:
            updated = with_first_playback(updated, ctx.clock.now())

    emitted_segments: list[TextSegmentDiagnostic] = []
    tts_segments: list[TextSegmentDiagnostic] = []
    final_buffer_flushed = False
    finish_reason: str | None = None

    async def on_text_submitted(item: TextWorkItem) -> None:
        tts_segments.append(TextSegmentDiagnostic(item.sequence, item.text))

    pipeline = SpeechPipeline(ctx.tts, ctx.playback, ctx.is_current, on_audio_generated, after_playback, on_text_submitted)

    async def produce(send: TextSender) -> None:
        nonlocal final_buffer_flushed, finish_reason, updated
        segmenter = IncrementalSpeechSegmenter()
        sequence = 0
        updated = TurnTimestamps(
            speech_end_detected=updated.speech_end_detected,
            stt_start=updated.stt_start,
            stt_invocation_start=updated.stt_invocation_start,
            stt_complete=updated.stt_complete,
            llm_invocation_start=updated.llm_invocation_start,
            llm_request_start=ctx.clock.now(),
        )
        async for event in ctx.llm.stream(messages, owned.cancellation):
            ctx.ensure_current(owned)
            if isinstance(event, TextChunk):
                assistant_parts.append(event.text)
                if updated.first_llm_token is None:
                    updated = with_first_llm_token(updated, ctx.clock.now())
                for segment in segmenter.push(normalize_spoken_text(event.text)):
                    emitted_segments.append(TextSegmentDiagnostic(sequence, segment))
                    updated = await _send_segment(ctx, owned, sequence, segment, send, updated)
                    sequence += 1
            elif isinstance(event, GenerationResult):
                finish_reason = event.status.value
                if event.metrics.first_sse_event_seconds is not None and updated.llm_request_start is not None:
                    updated = with_first_llm_raw_delta(updated, updated.llm_request_start + event.metrics.first_sse_event_seconds)
        for segment in segmenter.flush():
            final_buffer_flushed = True
            emitted_segments.append(TextSegmentDiagnostic(sequence, segment))
            updated = await _send_segment(ctx, owned, sequence, segment, send, updated)
            sequence += 1

    await pipeline.run_text_to_audio(produce, owned.cancellation)
    if owned.cancellation.is_cancelled():
        raise TurnCancelledError
    return updated, TextTurnDiagnostic("".join(assistant_parts), tuple(emitted_segments), tuple(tts_segments), final_buffer_flushed, finish_reason)


async def _send_segment(
    ctx: SpeakContext,
    owned: OwnedTurn,
    sequence: int,
    segment: str,
    send: TextSender,
    timestamps: TurnTimestamps,
) -> TurnTimestamps:
    ctx.ensure_current(owned)
    if ctx.get_state() is not RuntimeState.SPEAKING:
        ctx.move(RuntimeState.SPEAKING)
    await send(TextWorkItem(owned.turn_id, owned.generation_id, sequence, segment, ctx.style))
    return with_tts_request(timestamps, ctx.clock.now())
