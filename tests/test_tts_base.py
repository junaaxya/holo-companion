from collections.abc import AsyncIterator

import numpy as np
import pytest

from holo_companion.llm.base import CancellationHandle
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
    aggregate_audio_duration_seconds,
    real_time_factor,
)


class ImmediateTtsProvider:
    def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TTSAudioChunk | SynthesisResult]:
        return self._stream(request, cancellation)

    async def _stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None) -> AsyncIterator[TTSAudioChunk | SynthesisResult]:
        if cancellation is not None and cancellation.is_cancelled():
            raise TtsError(TtsErrorKind.CANCELLED, "pre-cancelled")

        audio_format = TtsAudioFormat(sample_rate_hz=4)
        first = TTSAudioChunk.from_samples(
            samples=np.ascontiguousarray(np.array([0.25, -0.25], dtype=np.float32)),
            audio_format=audio_format,
            sequence=0,
            start_seconds=0.0,
        )
        yield first

        if cancellation is not None and cancellation.is_cancelled():
            yield SynthesisResult(
                request=request,
                status=SynthesisStatus.CANCELLED,
                metrics=SynthesisMetrics(
                    request_started_at_seconds=0.0,
                    time_to_first_audio_seconds=0.2,
                    synthesis_seconds=0.4,
                    generated_audio_seconds=first.duration_seconds,
                    real_time_factor=real_time_factor(0.4, first.duration_seconds),
                    chunk_count=1,
                    received_byte_count=4,
                    provider="fake",
                    model="fake-model",
                ),
            )
            return

        second = TTSAudioChunk.from_samples(
            samples=np.ascontiguousarray(np.array([0.5, -0.5], dtype=np.float32)),
            audio_format=audio_format,
            sequence=1,
            start_seconds=first.duration_seconds,
        )
        yield second
        generated = aggregate_audio_duration_seconds((first, second))
        yield SynthesisResult(
            request=request,
            status=SynthesisStatus.COMPLETED,
            metrics=SynthesisMetrics(
                request_started_at_seconds=0.0,
                time_to_first_audio_seconds=0.2,
                synthesis_seconds=0.5,
                generated_audio_seconds=generated,
                real_time_factor=real_time_factor(0.5, generated),
                chunk_count=2,
                received_byte_count=8,
                provider="fake",
                model="fake-model",
            ),
        )


@pytest.mark.anyio
async def test_tts_provider_streams_first_audio_before_terminal_metrics() -> None:
    # Given
    provider = ImmediateTtsProvider()
    request = SynthesisRequest(text="Halo, ini suara pertama.")

    # When
    events = provider.stream(request)
    first = await anext(events)
    second = await anext(events)
    result = await anext(events)

    # Then
    assert isinstance(first, TTSAudioChunk)
    assert isinstance(second, TTSAudioChunk)
    assert first.sequence == 0
    assert first.start_seconds == 0.0
    assert first.audio_format == TtsAudioFormat(sample_rate_hz=4)
    assert first.duration_seconds == 0.5
    assert second.sequence == 1
    assert second.start_seconds == 0.5
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.COMPLETED
    assert result.metrics.request_started_at_seconds == 0.0
    assert result.metrics.time_to_first_audio_seconds == 0.2
    assert result.metrics.synthesis_seconds == 0.5
    assert result.metrics.generated_audio_seconds == 1.0
    assert result.metrics.real_time_factor == 0.5
    assert result.metrics.chunk_count == 2
    assert result.metrics.received_byte_count == 8
    assert result.metrics.provider == "fake"
    assert result.metrics.model == "fake-model"


@pytest.mark.anyio
async def test_tts_provider_raises_typed_error_when_cancelled_before_synthesis() -> None:
    # Given
    provider = ImmediateTtsProvider()
    cancellation = CancellationHandle()
    cancellation.cancel("barge-in")

    # When / Then
    with pytest.raises(TtsError) as raised:
        _ = [event async for event in provider.stream(SynthesisRequest(text="Halo"), cancellation)]
    assert raised.value.kind is TtsErrorKind.CANCELLED


@pytest.mark.anyio
async def test_tts_provider_returns_cancelled_terminal_metrics_after_first_chunk() -> None:
    # Given
    provider = ImmediateTtsProvider()
    cancellation = CancellationHandle()
    events = provider.stream(SynthesisRequest(text="Halo"), cancellation)

    # When
    first = await anext(events)
    cancellation.cancel("barge-in")
    result = await anext(events)

    # Then
    assert isinstance(first, TTSAudioChunk)
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.CANCELLED
    assert result.metrics.chunk_count == 1
    assert result.metrics.generated_audio_seconds == first.duration_seconds


def test_synthesis_request_preserves_raw_text_and_style_transport() -> None:
    # Given / When
    request = SynthesisRequest(text="  Halo\nworld  ", style=StyleHints(emotion="happy", energy=0.7, intensity=0.6, speaking_style="soft_playful"))

    # Then
    assert request.text == "  Halo\nworld  "
    assert request.style == StyleHints(emotion="happy", energy=0.7, intensity=0.6, speaking_style="soft_playful")


def test_synthesis_request_rejects_blank_text_and_invalid_style() -> None:
    # Given / When / Then
    with pytest.raises(TtsError, match="text"):
        SynthesisRequest(text=" \n\t ")
    with pytest.raises(TtsError, match="energy"):
        StyleHints(energy=1.1)
    with pytest.raises(TtsError, match="energy"):
        StyleHints(energy=float("nan"))
    with pytest.raises(TtsError, match="intensity"):
        StyleHints(intensity=1.1)
    with pytest.raises(TtsError, match="intensity"):
        StyleHints(intensity=float("nan"))
    with pytest.raises(TtsError, match="emotion"):
        StyleHints(emotion=" ")
    with pytest.raises(TtsError, match="speaking_style"):
        StyleHints(speaking_style=" ")


def test_tts_audio_format_requires_positive_native_mono_float32() -> None:
    # Given / When / Then
    assert TtsAudioFormat(sample_rate_hz=24_000).channels == 1
    assert TtsAudioFormat(sample_rate_hz=24_000).dtype == "float32"
    with pytest.raises(TtsError, match="sample_rate_hz"):
        TtsAudioFormat(sample_rate_hz=0)


@pytest.mark.parametrize(
    ("samples", "sequence", "start_seconds", "message"),
    [
        (np.ascontiguousarray(np.array([], dtype=np.float32)), 0, 0.0, "nonempty"),
        (np.ascontiguousarray(np.array([[0.0]], dtype=np.float32)), 0, 0.0, "rank 1"),
        (np.ascontiguousarray(np.array([0.0, 1.0, 2.0], dtype=np.float32))[::2], 0, 0.0, "contiguous"),
        (np.ascontiguousarray(np.array([0], dtype=np.int16)), 0, 0.0, "float32"),
        (np.ascontiguousarray(np.array([np.inf], dtype=np.float32)), 0, 0.0, "finite"),
        (np.ascontiguousarray(np.array([0.0], dtype=np.float32)), -1, 0.0, "sequence"),
        (np.ascontiguousarray(np.array([0.0], dtype=np.float32)), 0, -0.1, "start_seconds"),
    ],
)
def test_tts_audio_chunk_rejects_invalid_samples_sequence_and_start(samples: np.ndarray, sequence: int, start_seconds: float, message: str) -> None:
    # Given / When / Then
    with pytest.raises(TtsError, match=message):
        TTSAudioChunk.from_samples(samples=samples, audio_format=TtsAudioFormat(sample_rate_hz=16_000), sequence=sequence, start_seconds=start_seconds)


def test_aggregate_duration_and_rtf_handle_zero_audio() -> None:
    # Given
    audio_format = TtsAudioFormat(sample_rate_hz=10)
    first = TTSAudioChunk.from_samples(samples=np.ascontiguousarray(np.array([0.0, 0.1], dtype=np.float32)), audio_format=audio_format, sequence=0, start_seconds=0.0)

    # When / Then
    assert aggregate_audio_duration_seconds((first,)) == 0.2
    assert real_time_factor(0.5, 0.0) == 0.0


def test_typed_provider_error_propagates_kind_and_message() -> None:
    # Given / When
    error = TtsError(TtsErrorKind.PROVIDER, "provider refused synthesis")

    # Then
    assert error.kind is TtsErrorKind.PROVIDER
    assert str(error) == "provider: provider refused synthesis"
