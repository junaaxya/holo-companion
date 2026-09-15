from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import anyio
import numpy as np
import pytest

from holo_companion.llm.base import CancellationHandle
from holo_companion.runtime.pipeline import AUDIO_QUEUE_CAPACITY, TEXT_QUEUE_CAPACITY, AudioWorkItem, SpeechPipeline, TextWorkItem, _unpack_task_group_error
from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.tts.base import (
    StyleHints,
    SynthesisMetrics,
    SynthesisRequest,
    SynthesisResult,
    SynthesisStatus,
    TTSAudioChunk,
    TtsAudioFormat,
    TtsError,
    TtsErrorKind,
    TtsProviderDiagnostic,
    TtsStreamEvent,
)


def audio_chunk(sequence: int = 0) -> TTSAudioChunk:
    samples = np.ones(24, dtype=np.float32)
    return TTSAudioChunk.from_samples(samples, TtsAudioFormat(sample_rate_hz=24_000), sequence, 0.0)


def default_chunks() -> tuple[TTSAudioChunk, ...]:
    return (audio_chunk(),)


@dataclass(slots=True)
class RecordingTts:
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    requests: list[SynthesisRequest] = field(default_factory=list)
    chunks: tuple[TTSAudioChunk, ...] = field(default_factory=default_chunks)
    active: int = 0
    max_active: int = 0

    async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
        del cancellation
        self.entered.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.requests.append(request)
        try:
            if self.release is not None:
                await self.release.wait()
            for chunk in self.chunks:
                yield chunk
            yield SynthesisResult(request, SynthesisStatus.COMPLETED, SynthesisMetrics(0.0, 0.0, 0.0, 0.0, 0.0, len(self.chunks)))
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        return None


@dataclass(slots=True)
class RecordingPlayback:
    started: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event | None = None
    after_write: Callable[[], None] | None = None
    admissions: list[PlaybackAdmission] = field(default_factory=list)
    active: int = 0
    max_active: int = 0

    async def write_and_start(self, admission: PlaybackAdmission) -> None:
        self.started.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.admissions.append(admission)
        try:
            if self.release is not None:
                await self.release.wait()
            if self.after_write is not None:
                self.after_write()
        finally:
            self.active -= 1

    async def stop(self) -> None:
        if self.release is not None:
            self.release.set()

    async def aclose(self) -> None:
        return None

    async def wait_until_drained(self) -> None:
        pass

    async def set_generation_active(self, active: bool) -> None:
        del active


def current_guard(valid: bool = True) -> Callable[[int, int], bool]:
    def guard(turn_id: int, generation_id: int) -> bool:
        del turn_id, generation_id
        return valid

    return guard


@pytest.mark.anyio
async def test_pipeline_uses_bounded_text_and_audio_capacity_two() -> None:
    # Given
    pipeline = SpeechPipeline(tts=RecordingTts(), playback=RecordingPlayback(), is_current=current_guard())

    # When / Then
    assert pipeline.text_capacity == TEXT_QUEUE_CAPACITY == 2
    assert pipeline.audio_capacity == AUDIO_QUEUE_CAPACITY == 2


@pytest.mark.anyio
async def test_pipeline_backpressure_blocks_third_text_until_consumer_releases() -> None:
    # Given
    release = anyio.Event()
    tts = RecordingTts(release=release)
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())
    sent_third = anyio.Event()

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "satu", None))
        await send(TextWorkItem(1, 1, 1, "dua", None))
        await send(TextWorkItem(1, 1, 2, "tiga", None))
        sent_third.set()

    # When
    async with anyio.create_task_group() as tg:
        tg.start_soon(pipeline.run_text_to_audio, produce, CancellationHandle())
        await tts.entered.wait()
        assert not sent_third.is_set()
        release.set()

    # Then
    assert sent_third.is_set()
    assert [request.text for request in tts.requests] == ["satu", "dua", "tiga"]


@pytest.mark.anyio
async def test_pipeline_cancellation_releases_blocked_full_text_queue() -> None:
    # Given
    tts = RecordingTts(release=anyio.Event())
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())
    attempted_third = anyio.Event()

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "satu", None))
        await send(TextWorkItem(1, 1, 1, "dua", None))
        attempted_third.set()
        await send(TextWorkItem(1, 1, 2, "tiga", None))

    # When / Then
    with anyio.move_on_after(0.5) as timeout:
        async with anyio.create_task_group() as tg:
            tg.start_soon(pipeline.run_text_to_audio, produce, CancellationHandle())
            await attempted_third.wait()
            tg.cancel_scope.cancel()
    assert not timeout.cancelled_caught


@pytest.mark.anyio
async def test_pipeline_cancellation_releases_blocked_audio_playback_queue() -> None:
    # Given
    release = anyio.Event()
    playback = RecordingPlayback(release=release)
    tts = RecordingTts(chunks=(audio_chunk(0), audio_chunk(1), audio_chunk(2), audio_chunk(3)))
    pipeline = SpeechPipeline(tts=tts, playback=playback, is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "satu", None))

    # When / Then
    with anyio.move_on_after(0.5) as timeout:
        async with anyio.create_task_group() as tg:
            tg.start_soon(pipeline.run_text_to_audio, produce, CancellationHandle())
            await playback.started.wait()
            tg.cancel_scope.cancel()
            release.set()
    assert not timeout.cancelled_caught


@pytest.mark.anyio
async def test_pipeline_skips_stale_text_before_tts() -> None:
    # Given
    tts = RecordingTts()
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard(False))

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 0, 0, "stale", None))

    # When
    await pipeline.run_text_to_audio(produce, CancellationHandle())

    # Then
    assert tts.requests == []


@pytest.mark.anyio
async def test_pipeline_skips_stale_audio_before_and_after_playback() -> None:
    # Given
    valid = True

    def guard(turn_id: int, generation_id: int) -> bool:
        del turn_id, generation_id
        return valid

    def invalidate() -> None:
        nonlocal valid
        valid = False

    playback = RecordingPlayback(after_write=invalidate)
    pipeline = SpeechPipeline(tts=RecordingTts(), playback=playback, is_current=guard)

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "fresh", None))

    # When
    await pipeline.run_text_to_audio(produce, CancellationHandle())

    # Then
    assert len(playback.admissions) == 1
    assert pipeline.played_items == ()


@pytest.mark.anyio
async def test_pipeline_preserves_order_and_style_through_sequential_tts_and_sink() -> None:
    # Given
    style = StyleHints(emotion="happy", energy=0.4, intensity=0.3, speaking_style="soft")
    tts = RecordingTts()
    playback = RecordingPlayback()
    pipeline = SpeechPipeline(tts=tts, playback=playback, is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(3, 4, 0, "satu", style))
        await send(TextWorkItem(3, 4, 1, "dua", style))

    # When
    await pipeline.run_text_to_audio(produce, CancellationHandle())

    # Then
    assert [request.text for request in tts.requests] == ["satu", "dua"]
    assert [request.style for request in tts.requests] == [style, style]
    assert [admission.turn_id for admission in playback.admissions] == [3, 3]
    assert [item.sequence for item in pipeline.played_items] == [0, 1]
    assert tts.max_active == 1
    assert playback.max_active == 1


def test_pipeline_recovers_tts_error_nested_in_exception_group() -> None:
    # Given
    error = TtsError(TtsErrorKind.PROVIDER, "empty_audio")

    # When
    recovered = _unpack_task_group_error(ExceptionGroup("pipeline", [ExceptionGroup("synthesis", [error])]))

    # Then
    assert recovered is error


def test_pipeline_recovers_tts_error_with_broken_resource_cancellation_sibling() -> None:
    # Given
    error = TtsError(TtsErrorKind.PROVIDER, "empty_audio")

    # When
    recovered = _unpack_task_group_error(ExceptionGroup("pipeline", [error, anyio.BrokenResourceError()]))

    # Then
    assert recovered is error


def test_pipeline_preserves_unexpected_exception_group() -> None:
    # Given
    error = TtsError(TtsErrorKind.PROVIDER, "empty_audio")

    # When
    recovered = _unpack_task_group_error(ExceptionGroup("pipeline", [error, RuntimeError("unexpected")]))

    # Then
    assert recovered is None


@pytest.mark.anyio
async def test_pipeline_retries_empty_audio_once_and_succeeds() -> None:
    # Given
    empty_error = TtsError(
        TtsErrorKind.PROVIDER,
        "empty_audio",
        TtsProviderDiagnostic(stage="audio_decode", error="empty_audio"),
    )

    class EmptyThenSuccessTts(RecordingTts):
        attempt: int = 0

        async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
            self.requests.append(request)
            self.attempt += 1
            if self.attempt == 1:
                raise empty_error
            yield audio_chunk(0)

    tts = EmptyThenSuccessTts()
    playback = RecordingPlayback()
    pipeline = SpeechPipeline(tts=tts, playback=playback, is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "coba lagi", None))

    # When
    await pipeline.run_text_to_audio(produce, CancellationHandle())

    # Then
    assert tts.attempt == 2
    assert len(tts.requests) == 2
    assert len(playback.admissions) == 1


@pytest.mark.anyio
async def test_pipeline_does_not_retry_unrelated_tts_error() -> None:
    # Given
    timeout_error = TtsError(
        TtsErrorKind.PROVIDER,
        "timeout",
        TtsProviderDiagnostic(stage="timeout", error="timeout"),
    )

    class FailingTts(RecordingTts):
        async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
            self.requests.append(request)
            raise timeout_error
            yield audio_chunk(0)

    tts = FailingTts()
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "gagal", None))

    # When / Then
    with pytest.raises(TtsError) as exc_info:
        await pipeline.run_text_to_audio(produce, CancellationHandle())

    assert exc_info.value is timeout_error
    assert len(tts.requests) == 1


@pytest.mark.anyio
async def test_pipeline_does_not_retry_empty_audio_when_cancelled() -> None:
    # Given
    empty_error = TtsError(
        TtsErrorKind.PROVIDER,
        "empty_audio",
        TtsProviderDiagnostic(stage="audio_decode", error="empty_audio"),
    )

    class FailingTts(RecordingTts):
        async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
            self.requests.append(request)
            raise empty_error
            yield audio_chunk(0)

    tts = FailingTts()
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())
    cancellation = CancellationHandle()
    cancellation.cancel("barge-in")

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "batal", None))

    # When / Then
    with pytest.raises(TtsError) as exc_info:
        await pipeline.run_text_to_audio(produce, cancellation)

    assert exc_info.value is empty_error
    assert len(tts.requests) == 1


@pytest.mark.anyio
async def test_pipeline_logs_empty_audio_diagnostics_on_retry_and_final_failure(capsys) -> None:
    # Given
    empty_error = TtsError(
        TtsErrorKind.PROVIDER,
        "empty_audio",
        TtsProviderDiagnostic(
            stage="audio_decode",
            status_code=200,
            error="empty_audio",
            content_type="audio/pcm",
            received_byte_count=0,
            decoded_sample_count=0,
            stream_completed_normally=True,
        ),
    )

    class AlwaysEmptyTts(RecordingTts):
        async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
            self.requests.append(request)
            raise empty_error
            yield audio_chunk(0)

    tts = AlwaysEmptyTts()
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(5, 2, 3, "kopi hangat", None))

    # When / Then
    with pytest.raises(TtsError):
        await pipeline.run_text_to_audio(produce, CancellationHandle())

    assert len(tts.requests) == 2
    err = capsys.readouterr().err
    assert "turn_id: 5" in err
    assert "TTS segment index: 3" in err
    assert "segment text length: 11" in err
    assert "retry attempt: 0" in err
    assert "whether retry was triggered: True" in err
    assert "retry attempt: 1" in err
    assert "whether retry was triggered: False" in err
    assert "HTTP status: 200" in err
    assert "response content-type: audio/pcm" in err
    assert "whether HTTP stream completed normally: True" in err
