from collections.abc import AsyncIterator, Callable, Awaitable
import anyio
import numpy as np
import pytest

from holo_companion.llm.base import CancellationHandle
from holo_companion.runtime.pipeline import AudioWorkItem, SpeechPipeline, TextWorkItem
from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.runtime.state import RuntimeState
from holo_companion.runtime.turn import TurnErrorKind
from holo_companion.stt.base import Transcript, TranscriptSegment, Utterance
from holo_companion.tts.base import (
    SynthesisRequest,
    SynthesisResult,
    TTSAudioChunk,
    TtsAudioFormat,
    TtsError,
    TtsErrorKind,
    TtsProviderDiagnostic,
    TtsStreamEvent,
)
from runtime_fakes import FakeLlm, FakePlayback, FakeStt, FakeTts
from test_runtime_orchestrator import orchestrator, utterance, audio_chunk
from test_runtime_pipeline import RecordingTts, RecordingPlayback, current_guard


@pytest.mark.anyio
async def test_runtime_surfaces_elevenlabs_quota_exhausted_clearly() -> None:
    # Given
    quota_error = TtsError(
        TtsErrorKind.PROVIDER,
        "ElevenLabs quota exhausted",
        TtsProviderDiagnostic(stage="http_request", error="quota_exceeded", status_code=401),
    )

    class QuotaTts(FakeTts):
        async def stream(self, request, cancellation=None):
            self.requests.append(request)
            raise quota_error
            yield self.chunk

    runtime = orchestrator(tts=QuotaTts(audio_chunk()))
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None and result.error is not None
    assert result.error.kind is TurnErrorKind.PROVIDER
    assert result.error.stage == "tts"
    diag = result.error.diagnostic
    assert diag is not None
    assert diag.provider == "elevenlabs"
    assert diag.provider_code == "quota_exceeded"
    assert diag.message == "ElevenLabs quota exhausted"
    assert diag.http_status == 401
    assert diag.recoverable is True


@pytest.mark.anyio
async def test_pipeline_does_not_retry_quota_exceeded() -> None:
    # Given
    quota_error = TtsError(
        TtsErrorKind.PROVIDER,
        "ElevenLabs quota exhausted",
        TtsProviderDiagnostic(stage="http_request", error="quota_exceeded", status_code=401),
    )

    class FailingTts(RecordingTts):
        async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
            self.requests.append(request)
            raise quota_error
            yield audio_chunk(0)

    tts = FailingTts()
    pipeline = SpeechPipeline(tts=tts, playback=RecordingPlayback(), is_current=current_guard())

    async def produce(send: Callable[[TextWorkItem], Awaitable[None]]) -> None:
        await send(TextWorkItem(1, 1, 0, "halo", None))

    # When / Then
    with pytest.raises(TtsError) as exc_info:
        await pipeline.run_text_to_audio(produce, CancellationHandle())

    assert exc_info.value is quota_error
    assert len(tts.requests) == 1
