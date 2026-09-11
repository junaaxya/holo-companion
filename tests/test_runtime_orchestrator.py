import anyio
import numpy as np
import pytest

from holo_companion.runtime.metrics import ManualClock
from holo_companion.runtime.orchestrator import RuntimeOrchestrator
from holo_companion.runtime.state import InvalidStateTransition, RuntimeState, transition
from holo_companion.runtime.turn import TurnErrorKind
from holo_companion.stt.base import Transcript, TranscriptSegment, Utterance
from holo_companion.tts.base import StyleHints, TTSAudioChunk, TtsAudioFormat
from runtime_fakes import FakeLlm, FakePlayback, FakeStt, FakeTts


def utterance() -> Utterance:
    samples = np.ones(160, dtype=np.float32)
    return Utterance.from_samples(samples)


def transcript(text: str = "halo holo") -> Transcript:
    return Transcript(
        text=text,
        language="id",
        language_probability=1.0,
        audio_duration_seconds=0.01,
        processing_seconds=0.02,
        real_time_factor=2.0,
        model_name="fake",
        device="cpu",
        compute_type="int8",
        segments=(TranscriptSegment(0.0, 0.01, text),),
    )


def audio_chunk(sequence: int = 0) -> TTSAudioChunk:
    samples = np.ones(24, dtype=np.float32)
    return TTSAudioChunk.from_samples(samples, TtsAudioFormat(sample_rate_hz=24_000), sequence, 0.0)


def orchestrator(
    *,
    stt: FakeStt | None = None,
    llm: FakeLlm | None = None,
    tts: FakeTts | None = None,
    playback: FakePlayback | None = None,
    clock: ManualClock | None = None,
) -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        stt=stt or FakeStt(transcript()),
        llm=llm or FakeLlm(),
        tts=tts or FakeTts(audio_chunk()),
        playback=playback or FakePlayback(),
        clock=clock or ManualClock(),
    )


@pytest.mark.anyio
async def test_normal_turn_records_sequence_and_nullable_metrics_when_successful() -> None:
    clock = ManualClock()
    runtime = orchestrator(clock=clock)
    await runtime.start()
    clock.advance(0.10)
    await runtime.speech_started()
    clock.advance(0.20)
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    assert runtime.transition_history == (
        RuntimeState.IDLE,
        RuntimeState.LISTENING,
        RuntimeState.TRANSCRIBING,
        RuntimeState.THINKING,
        RuntimeState.SPEAKING,
        RuntimeState.IDLE,
        RuntimeState.STOPPED,
    )
    assert result is not None
    assert result.turn_id == 1
    assert result.generation_id == 1
    assert result.metrics.speech_end_to_transcript_seconds == pytest.approx(0.0)
    assert result.metrics.speech_end_to_first_playback_seconds == pytest.approx(0.0)
    assert result.error is None
    assert result.utterance_sample_count == utterance().samples.size
    assert result.utterance_peak == pytest.approx(1.0)
    assert result.utterance_rms == pytest.approx(1.0)


def test_invalid_transition_raises_typed_error() -> None:
    with pytest.raises(InvalidStateTransition) as error:
        transition(RuntimeState.IDLE, RuntimeState.SPEAKING)
    assert error.value.from_state is RuntimeState.IDLE
    assert error.value.to_state is RuntimeState.SPEAKING


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["stt", "llm", "tts"])
async def test_provider_failure_enters_error_and_preserves_metrics(stage: str) -> None:
    runtime = orchestrator(stt=FakeStt(transcript(), fail=stage == "stt"), llm=FakeLlm(fail=stage == "llm"), tts=FakeTts(audio_chunk(), fail=stage == "tts"))
    await runtime.start()
    await runtime.speech_started()
    result = await runtime.speech_ended(utterance())
    await runtime.recover_from_error()
    await runtime.aclose()

    assert result is not None
    assert result.error is not None
    assert result.error.kind is TurnErrorKind.PROVIDER
    assert result.error.stage == stage
    assert RuntimeState.ERROR in runtime.transition_history
    assert runtime.state is RuntimeState.STOPPED
    assert result.metrics.speech_end_detected is not None


@pytest.mark.anyio
async def test_recoverable_provider_failure_detaches_turn_and_next_speech_is_accepted() -> None:
    # Given
    llm = FakeLlm(fail=True)
    runtime = orchestrator(llm=llm)
    await runtime.start()
    await runtime.speech_started()

    # When
    failed = await runtime.speech_ended(utterance())
    llm.fail = False
    await runtime.speech_started()
    fresh = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert failed is not None and failed.error is not None
    assert failed.error.stage == "llm"
    assert failed.turn_id == 1
    assert fresh is not None and fresh.error is None
    assert fresh.turn_id == 2
    assert RuntimeState.ERROR in runtime.transition_history
    assert runtime.transition_history.count(RuntimeState.LISTENING) == 2


@pytest.mark.anyio
async def test_llm_error_preserves_allowlisted_diagnostic_without_secret_message() -> None:
    # Given
    runtime = orchestrator(llm=FakeLlm(fail=True))
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None and result.error is not None
    diagnostic = result.error.diagnostic
    assert diagnostic is not None
    assert diagnostic.provider == "openai-compatible"
    assert diagnostic.exception_class == "LlmError"
    assert diagnostic.message == "protocol"
    assert diagnostic.http_status is None
    assert diagnostic.recoverable is True
    assert "secret" not in repr(diagnostic)


@pytest.mark.anyio
async def test_unknown_programming_error_propagates_without_error_state_result() -> None:
    runtime = orchestrator(llm=FakeLlm(bug=True))
    await runtime.start()
    await runtime.speech_started()

    with pytest.raises(ValueError, match="llm programming bug"):
        await runtime.speech_ended(utterance())

    assert not runtime.turn_results
    await runtime.aclose()


@pytest.mark.anyio
async def test_shutdown_active_turn_cancels_stops_joins_and_closes_owned_resources() -> None:
    release = anyio.Event()
    stt = FakeStt(transcript(), release=release)
    llm = FakeLlm()
    tts = FakeTts(audio_chunk())
    playback = FakePlayback()
    runtime = orchestrator(stt=stt, llm=llm, tts=tts, playback=playback)

    await runtime.start()
    await runtime.speech_started()
    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await stt.entered.wait()
        await runtime.aclose()
        release.set()
        tg.cancel_scope.cancel()

    assert stt.max_active == 1
    assert stt.closed
    assert llm.closed
    assert tts.closed
    assert playback.closed
    assert runtime.state is RuntimeState.STOPPED


@pytest.mark.anyio
async def test_stale_waiting_turn_never_invokes_stt_after_newest_turn_supersedes_it() -> None:
    # Given
    release = anyio.Event()
    stt = FakeStt(transcript(), release=release)
    runtime = orchestrator(stt=stt)
    await runtime.start()
    await runtime.speech_started()

    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await stt.entered.wait()
        await runtime.speech_started()
        tg.start_soon(runtime.speech_ended, utterance())
        await anyio.lowlevel.checkpoint()
        await runtime.speech_started()
        release.set()
        newest = await runtime.speech_ended(utterance())

    # Then
    await runtime.aclose()
    assert stt.max_active == 1
    assert newest is not None and newest.turn_id == 3
    assert [result.turn_id for result in runtime.turn_results if result.error is None] == [3]
    assert runtime.stt_diagnostics.cancelled_before_start >= 1
    assert runtime.stt_diagnostics.stale_results_discarded >= 1


@pytest.mark.anyio
async def test_new_speech_while_speaking_stops_old_playback_before_replacement() -> None:
    release = anyio.Event()
    playback = FakePlayback(release=release)
    runtime = orchestrator(playback=playback)
    await runtime.start()
    await runtime.speech_started()

    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await playback.started.wait()
        await runtime.speech_started()
        await runtime.speech_ended(utterance())

    await runtime.aclose()
    assert playback.stopped.is_set()
    assert [admission.turn_id for admission in playback.admissions] == [1, 2]
    assert playback.max_active == 1


@pytest.mark.anyio
async def test_mid_llm_cancellation_releases_stream_and_no_stale_audio_plays() -> None:
    release = anyio.Event()
    llm = FakeLlm(release=release)
    playback = FakePlayback()
    runtime = orchestrator(llm=llm, playback=playback)
    await runtime.start()
    await runtime.speech_started()

    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await llm.entered.wait()
        await runtime.speech_started()
        release.set()
        await runtime.speech_ended(utterance())

    await runtime.aclose()
    assert llm.cancelled
    assert [admission.turn_id for admission in playback.admissions] == [2]


@pytest.mark.anyio
async def test_interruption_cancels_owned_assistant_task_without_caller_cancel() -> None:
    release = anyio.Event()
    llm = FakeLlm(release=release)
    playback = FakePlayback()
    runtime = orchestrator(llm=llm, playback=playback)
    await runtime.start()
    await runtime.speech_started()

    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await llm.entered.wait()
        await runtime.speech_started()
        release.set()
        await runtime.speech_ended(utterance())

    await runtime.aclose()
    assert llm.cancelled
    assert [result.turn_id for result in runtime.turn_results] == [1, 2]
    assert runtime.turn_results[0].error is not None
    assert runtime.turn_results[0].error.kind is TurnErrorKind.CANCELLED


@pytest.mark.anyio
async def test_mid_tts_cancellation_releases_stream_and_no_stale_audio_plays() -> None:
    release = anyio.Event()
    tts = FakeTts(audio_chunk(), release=release)
    playback = FakePlayback()
    runtime = orchestrator(tts=tts, playback=playback)
    await runtime.start()
    await runtime.speech_started()

    async with anyio.create_task_group() as tg:
        tg.start_soon(runtime.speech_ended, utterance())
        await tts.entered.wait()
        await runtime.speech_started()
        release.set()
        await runtime.speech_ended(utterance())

    await runtime.aclose()
    assert tts.cancelled
    assert [admission.turn_id for admission in playback.admissions] == [2]


@pytest.mark.anyio
async def test_stale_generation_rejected_immediately_before_sink_admission() -> None:
    runtime = orchestrator()
    await runtime.start()
    await runtime.speech_started()
    await runtime.reject_stale_for_test(turn_id=1, generation_id=0, chunk=audio_chunk())
    await runtime.aclose()

    assert runtime.turn_results[0].error is not None
    assert runtime.turn_results[0].error.kind is TurnErrorKind.STALE_OUTPUT


@pytest.mark.anyio
async def test_functional_contract_fakes_run_existing_provider_shapes_end_to_end() -> None:
    playback = FakePlayback()
    runtime = orchestrator(stt=FakeStt(result=transcript("apa kabar")), llm=FakeLlm(chunks=("baik", " banget")), playback=playback)
    await runtime.start()
    await runtime.speech_started()
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    assert result is not None
    assert result.transcript == "apa kabar"
    assert result.assistant_text == "baik banget"
    assert len(playback.admissions) == 1
    assert [admission.turn_id for admission in playback.admissions] == [1]


@pytest.mark.anyio
async def test_segmented_llm_text_is_sent_to_tts_with_ordered_style_and_flush() -> None:
    # Given
    style = StyleHints(emotion="happy", energy=0.5, intensity=0.4, speaking_style="soft")
    tts = FakeTts(audio_chunk())
    runtime = RuntimeOrchestrator(stt=FakeStt(transcript()), llm=FakeLlm(chunks=("Halo", ", Holo.", " Apa", " kabar?", " Sisa")), tts=tts, playback=FakePlayback(), clock=ManualClock(), style=style)
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None
    assert result.assistant_text == "Halo, Holo. Apa kabar? Sisa"
    assert [request.text for request in tts.requests] == ["Halo, Holo.", " Apa kabar?", " Sisa"]
    assert [request.style for request in tts.requests] == [style, style, style]


@pytest.mark.anyio
async def test_first_timestamp_metrics_are_set_only_once_across_multiple_segments() -> None:
    # Given
    clock = ManualClock()
    tts = FakeTts(audio_chunk())
    runtime = orchestrator(llm=FakeLlm(chunks=("Satu.", " Dua.")), tts=tts, clock=clock)
    await runtime.start()
    await runtime.speech_started()
    clock.advance(1.0)

    # When
    result = await runtime.speech_ended(utterance())
    clock.advance(10.0)
    await runtime.aclose()

    # Then
    assert result is not None
    assert result.metrics.first_llm_token == pytest.approx(1.0)
    assert result.metrics.tts_request_start == pytest.approx(1.0)
    assert result.metrics.first_tts_audio == pytest.approx(1.0)
    assert result.metrics.first_playback_audio == pytest.approx(1.0)


@pytest.mark.anyio
async def test_runtime_records_provider_raw_and_usable_llm_timing_separately() -> None:
    # Given
    clock = ManualClock()
    runtime = orchestrator(llm=FakeLlm(chunks=("Halo.",)), clock=clock)
    await runtime.start()
    await runtime.speech_started()
    clock.advance(1.0)

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None
    assert result.metrics.llm_invocation_start == pytest.approx(1.0)
    assert result.metrics.llm_request_start == pytest.approx(1.0)
    assert result.metrics.first_llm_raw_delta == pytest.approx(1.01)
    assert result.metrics.first_llm_token == pytest.approx(1.0)
    assert result.metrics.llm_provider_request_to_first_raw_delta_seconds == pytest.approx(0.01)
    assert result.metrics.llm_provider_request_to_first_usable_delta_seconds == pytest.approx(0.0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("chunks", "expected_segments", "flushed"),
    [
        (("Seduh kopi", " dengan tenang"), ("Seduh kopi dengan tenang",), True),
        (("Seduh kopi, kacang tanah,", " lalu santai."), ("Seduh kopi, kacang tanah, lalu santai.",), False),
        (("Satu.", " Dua?", " Tiga tanpa akhir"), ("Satu.", " Dua?", " Tiga tanpa akhir"), True),
    ],
)
async def test_text_diagnostic_preserves_every_segment_to_tts(
    chunks: tuple[str, ...], expected_segments: tuple[str, ...], flushed: bool
) -> None:
    # Given
    tts = FakeTts(audio_chunk())
    runtime = orchestrator(llm=FakeLlm(chunks=chunks), tts=tts)
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None and result.text_diagnostic is not None
    diagnostic = result.text_diagnostic
    assert diagnostic.llm_text == "".join(chunks)
    assert tuple(segment.text for segment in diagnostic.emitted_segments) == expected_segments
    assert tuple(segment.text for segment in diagnostic.tts_segments) == expected_segments
    assert tuple(segment.index for segment in diagnostic.emitted_segments) == tuple(range(len(expected_segments)))
    assert tuple(segment.index for segment in diagnostic.tts_segments) == tuple(range(len(expected_segments)))
    assert diagnostic.final_buffer_flushed is flushed
    assert diagnostic.llm_finish_reason == "completed"
    assert [request.text for request in tts.requests] == list(expected_segments)


@pytest.mark.anyio
async def test_spoken_text_normalizes_markdown_and_merges_short_list_fragments() -> None:
    # Given
    tts = FakeTts(audio_chunk())
    runtime = orchestrator(llm=FakeLlm(chunks=("## Makanan:\n1.", " **Seduh kopi** dengan tenang.", "\n2. Kacang tanah cocok.")), tts=tts)
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None and result.text_diagnostic is not None
    segments = tuple(segment.text for segment in result.text_diagnostic.tts_segments)
    assert segments == ("Makanan: Seduh kopi dengan tenang.", "Kacang tanah cocok.")
    assert all(segment.strip() not in {"1.", "2.", "Makanan:"} for segment in segments)




@pytest.mark.anyio
async def test_spoken_text_keeps_comma_heavy_stream_as_natural_sentence() -> None:
    # Given
    tts = FakeTts(audio_chunk())
    runtime = orchestrator(llm=FakeLlm(chunks=("Kalau mau, kita bisa", " seduh kopi, makan kacang tanah, lalu santai.")), tts=tts)
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None and result.text_diagnostic is not None
    assert tuple(segment.text for segment in result.text_diagnostic.tts_segments) == ("Kalau mau, kita bisa seduh kopi, makan kacang tanah, lalu santai.",)


@pytest.mark.anyio
async def test_normal_assistant_replies_play_to_completion_with_playback_sink() -> None:
    # Given
    tts = FakeTts(audio_chunk())
    playback = FakePlayback()
    runtime = orchestrator(llm=FakeLlm(chunks=("Halo, Master.", " Ada yang bisa dibantu?")), tts=tts, playback=playback)
    await runtime.start()
    await runtime.speech_started()

    # When
    result = await runtime.speech_ended(utterance())
    await runtime.aclose()

    # Then
    assert result is not None
    assert result.error is None
    assert result.assistant_text == "Halo, Master. Ada yang bisa dibantu?"
    assert len(playback.admissions) == 2
    assert [adm.turn_id for adm in playback.admissions] == [1, 1]
    assert playback.closed is True


