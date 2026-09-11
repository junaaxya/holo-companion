import anyio

from holo_companion.llm.base import CancellationHandle, ChatMessage, LLMProvider, LlmError
from holo_companion.runtime._speak import SpeakContext, generate_and_speak
from holo_companion.runtime.metrics import MonotonicClock, SystemMonotonicClock, TurnTimestamps
from holo_companion.runtime.ownership import OwnedTurn, StaleOutputError, TurnCancelledError, ensure_current, is_current
from holo_companion.runtime.playback import PlaybackAdmission, PlaybackSink
from holo_companion.runtime.provider_diagnostics import provider_diagnostic
from holo_companion.runtime.result_factory import turn_result
from holo_companion.runtime.state import RuntimeState, transition
from holo_companion.runtime.stt_admission import SttAdmissionDiagnostics, bump_stt_cancel_diagnostic
from holo_companion.runtime.turn import InterruptionDiagnostic, TextTurnDiagnostic, TurnError, TurnErrorKind, TurnResult
from holo_companion.stt.base import STTProvider, SttError, Transcript, Utterance
from holo_companion.tts.base import StyleHints, TTSAudioChunk, TTSProvider, TtsError
class RuntimeOrchestrator:
    def __init__(
        self,
        *,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        playback: PlaybackSink,
        clock: MonotonicClock | None = None,
        style: StyleHints | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._playback = playback
        self._clock = clock or SystemMonotonicClock()
        self._style = style or StyleHints(emotion="neutral", energy=0.5, intensity=0.5, speaking_style="neutral")
        self._state = RuntimeState.IDLE
        self._history: list[RuntimeState] = [RuntimeState.IDLE]
        self._results: list[TurnResult] = []
        self._turn_id = 0
        self._generation_id = 0
        self._active: OwnedTurn | None = None
        self._started = False
        self._lock = anyio.Lock()
        self._stt_limiter = anyio.CapacityLimiter(1)
        self._stt_diagnostics = SttAdmissionDiagnostics()
    @property
    def state(self) -> RuntimeState:
        return self._state
    @property
    def transition_history(self) -> tuple[RuntimeState, ...]:
        return tuple(self._history)
    @property
    def turn_results(self) -> tuple[TurnResult, ...]:
        return tuple(self._results)

    @property
    def stt_diagnostics(self) -> SttAdmissionDiagnostics:
        return self._stt_diagnostics

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True

    async def speech_started(self, start_frame_index: int | None = None) -> None:
        await self.start()
        async with self._lock:
            if self._active is not None:
                await self._interrupt_locked(start_frame_index)
            self._turn_id += 1
            self._generation_id += 1
            self._active = OwnedTurn(self._turn_id, self._generation_id, CancellationHandle())
            self._move(RuntimeState.LISTENING)

    async def speech_ended(self, utterance: Utterance) -> TurnResult | None:
        owned = self._active
        if owned is None:
            return None
        timestamps = TurnTimestamps(speech_end_detected=self._clock.now())
        transcript: Transcript | None = None
        assistant_parts: list[str] = []
        text_diagnostic: TextTurnDiagnostic | None = None
        try:
            with anyio.CancelScope() as scope:
                owned.scope = scope
                timestamps, transcript = await self._transcribe(owned, utterance, timestamps)
                messages = (ChatMessage("user", transcript.text),)
                ctx = SpeakContext(
                    llm=self._llm, tts=self._tts, playback=self._playback,
                    clock=self._clock, style=self._style,
                    move=self._move, ensure_current=self._ensure_current,
                    is_current=self._is_current, get_state=lambda: self._state,
                )
                timestamps, text_diagnostic = await generate_and_speak(ctx, owned, messages, assistant_parts, timestamps)
                owned.scope = None
        except TurnCancelledError:
            return self._finish(owned, transcript, utterance, assistant_parts, timestamps, TurnError(TurnErrorKind.CANCELLED, "turn", interruption=owned.interruption), text_diagnostic)
        except StaleOutputError:
            return self._finish(owned, transcript, utterance, assistant_parts, timestamps, TurnError(TurnErrorKind.STALE_OUTPUT, "playback"))
        except SttError as error:
            return await self._recover_provider_error(owned, transcript, utterance, assistant_parts, timestamps, "stt", error)
        except LlmError as error:
            return await self._recover_provider_error(owned, transcript, utterance, assistant_parts, timestamps, "llm", error)
        except TtsError as error:
            return await self._recover_provider_error(owned, transcript, utterance, assistant_parts, timestamps, "tts", error)
        # CancelScope suppresses its own CancelledError; check explicitly after.
        if scope.cancelled_caught or owned.cancellation.is_cancelled():
            self._stt_diagnostics = bump_stt_cancel_diagnostic(
                self._stt_diagnostics, stt_ran=owned.stt_started
            )
            return self._finish(owned, transcript, utterance, assistant_parts, timestamps, TurnError(TurnErrorKind.CANCELLED, "turn", interruption=owned.interruption))
        return self._finish(owned, transcript, utterance, assistant_parts, timestamps, None, text_diagnostic)

    async def reject_stale_for_test(self, *, turn_id: int, generation_id: int, chunk: TTSAudioChunk) -> None:
        if not self._is_current(turn_id, generation_id):
            self._results.append(TurnResult(turn_id, generation_id, None, "", TurnTimestamps(), TurnError(TurnErrorKind.STALE_OUTPUT, "playback")))
            return
        await self._playback.write_and_start(PlaybackAdmission(turn_id, generation_id, chunk))
        if not self._is_current(turn_id, generation_id):
            self._results.append(TurnResult(turn_id, generation_id, None, "", TurnTimestamps(), TurnError(TurnErrorKind.STALE_OUTPUT, "playback")))

    async def recover_from_error(self) -> None:
        async with self._lock:
            if self._state is RuntimeState.ERROR:
                self._move(RuntimeState.IDLE)

    async def aclose(self) -> None:
        async with self._lock:
            if self._state is RuntimeState.STOPPED:
                return
            if self._active is not None:
                self._active.cancellation.cancel("runtime shutdown")
                if self._active.scope is not None:
                    self._active.scope.cancel()
            await self._playback.stop()
            await self._tts.aclose()
            await self._llm.aclose()
            stt_close = getattr(self._stt, "aclose", None)
            if stt_close is not None:
                await stt_close()
            await self._playback.aclose()
            if self._state is not RuntimeState.STOPPED:
                self._move(RuntimeState.STOPPED)

    async def _transcribe(self, owned: OwnedTurn, utterance: Utterance, timestamps: TurnTimestamps) -> tuple[TurnTimestamps, Transcript]:
        self._move(RuntimeState.TRANSCRIBING)
        invocation: float | None = None

        def transcribe() -> Transcript:
            nonlocal invocation
            invocation = self._clock.now()
            owned.stt_started = True
            self._stt_diagnostics = SttAdmissionDiagnostics(
                self._stt_diagnostics.active_inferences + 1,
                self._stt_diagnostics.cancelled_before_start,
                self._stt_diagnostics.stale_results_discarded,
            )
            try:
                return self._stt.transcribe(utterance, None)
            finally:
                self._stt_diagnostics = SttAdmissionDiagnostics(
                    self._stt_diagnostics.active_inferences - 1,
                    self._stt_diagnostics.cancelled_before_start,
                    self._stt_diagnostics.stale_results_discarded,
                )

        self._ensure_current(owned)
        transcript = await anyio.to_thread.run_sync(transcribe, abandon_on_cancel=False, limiter=self._stt_limiter)
        try:
            self._ensure_current(owned)
        except (TurnCancelledError, StaleOutputError):
            self._stt_diagnostics = SttAdmissionDiagnostics(
                self._stt_diagnostics.active_inferences,
                self._stt_diagnostics.cancelled_before_start,
                self._stt_diagnostics.stale_results_discarded + 1,
            )
            raise
        return TurnTimestamps(
            speech_end_detected=timestamps.speech_end_detected,
            stt_start=invocation,
            stt_invocation_start=invocation,
            stt_complete=self._clock.now(),
        ), transcript

    async def _interrupt_locked(self, start_frame_index: int | None = None) -> None:
        active = self._active
        self._active = None
        previous_state = self._state.value.lower()
        self._move(RuntimeState.INTERRUPTING)
        if active is not None:
            active.interruption = InterruptionDiagnostic(
                previous_turn_id=active.turn_id,
                previous_state=previous_state,
                new_speech_frame=start_frame_index,
                reason="barge_in",
            )
            active.cancellation.cancel("interrupted by new speech")
            if active.scope is not None:
                active.scope.cancel()
        await self._playback.stop()

    async def _recover_provider_error(
        self,
        owned: OwnedTurn,
        transcript: Transcript | None,
        utterance: Utterance,
        assistant_parts: list[str],
        timestamps: TurnTimestamps,
        stage: str,
        error: SttError | LlmError | TtsError,
    ) -> TurnResult:
        if self._is_current(owned.turn_id, owned.generation_id):
            self._active = None
            self._move(RuntimeState.ERROR)
            await self._playback.stop()
            self._move(RuntimeState.IDLE)
        result = turn_result(
            owned, transcript, utterance, "".join(assistant_parts), timestamps,
            TurnError(TurnErrorKind.PROVIDER, stage, provider_diagnostic(error)),
        )
        self._results.append(result)
        return result

    def _finish(
        self,
        owned: OwnedTurn,
        transcript: Transcript | None,
        utterance: Utterance,
        assistant_parts: list[str],
        timestamps: TurnTimestamps,
        error: TurnError | None,
        text_diagnostic: TextTurnDiagnostic | None = None,
    ) -> TurnResult:
        if self._is_current(owned.turn_id, owned.generation_id):
            self._active = None
            if error is None:
                self._move(RuntimeState.IDLE)
        result = turn_result(owned, transcript, utterance, "".join(assistant_parts), timestamps, error, text_diagnostic)
        self._results.append(result)
        return result

    def _ensure_current(self, owned: OwnedTurn) -> None:
        ensure_current(self._active, owned)

    def _is_current(self, turn_id: int, generation_id: int) -> bool:
        return is_current(self._active, turn_id, generation_id)

    def _move(self, to_state: RuntimeState) -> None:
        self._state = transition(self._state, to_state)
        self._history.append(to_state)
