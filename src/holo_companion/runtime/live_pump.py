import threading
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, assert_never

import anyio
from anyio import WouldBlock
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from holo_companion.audio.capture import AudioBackend
from holo_companion.audio.raw_capture import RawCaptureDumper
from holo_companion.audio.types import CAPTURE_CHANNELS, CAPTURE_DTYPE, CAPTURE_FRAME_SAMPLES, CAPTURE_SAMPLE_RATE, AudioFrame, DeviceIndex
from holo_companion.runtime.barge_in import AdaptiveBargeInGate, GateRejection
from holo_companion.runtime.events import SpeechEnded, SpeechStarted
from holo_companion.runtime.state import RuntimeState
from holo_companion.stt.base import Utterance
from holo_companion.stt.collector import UtteranceCollector
from holo_companion.vad.base import TurnDetector, VadProvider
from holo_companion.runtime.turn import TurnResult

DEFAULT_CAPTURE_BUFFER_CAPACITY = 8
DEFAULT_INTERRUPTION_GUARD_MS = 300
ACTIVE_BARGE_IN_STATES = frozenset((RuntimeState.TRANSCRIBING, RuntimeState.THINKING, RuntimeState.SPEAKING))


class LiveOrchestrator(Protocol):
    @property
    def state(self) -> RuntimeState: ...

    async def speech_started(self, start_frame_index: int | None = None) -> None: ...

    async def speech_ended(self, utterance: Utterance) -> TurnResult | None: ...


class UtteranceDumper(Protocol):
    def __call__(self, utterance: Utterance) -> None: ...


class FloatClock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class LiveDiagnostics:
    capture_overruns: int = 0
    input_underruns: int = 0
    self_trigger_suppressions: int = 0
    barge_in_accepted: int = 0
    barge_in_rejected_noise: int = 0
    barge_in_rejected_short: int = 0
    barge_in_echo_correlation: float | None = None
    barge_in_rejected_echo: int = 0


@dataclass(slots=True)
class InterruptionGuard:
    guard_ms: int = DEFAULT_INTERRUPTION_GUARD_MS
    monotonic_seconds: FloatClock | None = None
    playback_consumed_at_seconds: float | None = None

    def mark_playback_consumed(self) -> None:
        self.playback_consumed_at_seconds = self._now()

    def should_suppress(self) -> bool:
        if self.playback_consumed_at_seconds is None or self.guard_ms <= 0:
            return False
        elapsed_ms = (self._now() - self.playback_consumed_at_seconds) * 1000
        return 0 <= elapsed_ms <= self.guard_ms

    def _now(self) -> float:
        if self.monotonic_seconds is None:
            return anyio.current_time()
        return self.monotonic_seconds()


@dataclass(slots=True)
class CaptureVadPump:
    backend: AudioBackend
    vad: VadProvider
    detector: TurnDetector
    collector: UtteranceCollector
    orchestrator: LiveOrchestrator
    input_device: DeviceIndex
    interruption_guard: InterruptionGuard | None = None
    frame_capacity: int = DEFAULT_CAPTURE_BUFFER_CAPACITY
    diagnostics: LiveDiagnostics = LiveDiagnostics()
    on_turn_complete: Callable[[TurnResult], Awaitable[None]] | None = None
    dump_utterance: UtteranceDumper | None = None
    raw_capture_dumper: RawCaptureDumper | None = None
    barge_in_gate: AdaptiveBargeInGate = field(default_factory=AdaptiveBargeInGate)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    async def run(self) -> LiveDiagnostics:
        send, receive = anyio.create_memory_object_stream[AudioFrame](max_buffer_size=self.frame_capacity)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._capture, send)
                await self._process(receive, tg)
                tg.cancel_scope.cancel()
        finally:
            if self.raw_capture_dumper is not None:
                self.raw_capture_dumper.close()
        return self.diagnostics

    async def _capture(self, send: MemoryObjectSendStream[AudioFrame]) -> None:
        stream = self.backend.frame_stream(
            samplerate=int(CAPTURE_SAMPLE_RATE),
            channels=CAPTURE_CHANNELS,
            dtype=CAPTURE_DTYPE,
            device=self.input_device,
            blocksize=CAPTURE_FRAME_SAMPLES,
        )
        try:
            await anyio.to_thread.run_sync(self._drain_blocking_stream, stream, send, abandon_on_cancel=True)
        finally:
            self._stop.set()

    def _drain_blocking_stream(self, stream: Iterator[AudioFrame], send: MemoryObjectSendStream[AudioFrame]) -> None:
        with send:
            for audio_frame in stream:
                if self._stop.is_set():
                    break
                try:
                    anyio.from_thread.run_sync(send.send_nowait, audio_frame)
                except WouldBlock:
                    self._count_capture_overrun()
                except RuntimeError:
                    break

    async def _process(self, receive: MemoryObjectReceiveStream[AudioFrame], tg: anyio.abc.TaskGroup) -> None:
        async with receive:
            async for audio_frame in receive:
                if self.raw_capture_dumper is not None:
                    self.raw_capture_dumper.write_frame(audio_frame)
                probability = self.vad.process(audio_frame)
                if self._is_verified_nonspeech(probability):
                    self.barge_in_gate.observe_verified_nonspeech(audio_frame)
                events = list(self.detector.process(audio_frame, probability))
                accepted = await self._dispatch_events(events, audio_frame, probability)
                utterance = self.collector.process(audio_frame, accepted)
                if utterance is not None:
                    if self.dump_utterance is not None:
                        self.dump_utterance(utterance)
                    tg.start_soon(self._run_turn, utterance)

    async def _run_turn(self, utterance: Utterance) -> None:
        result = await self.orchestrator.speech_ended(utterance)
        if result is not None and self.on_turn_complete is not None:
            await self.on_turn_complete(result)

    async def _dispatch_events(self, events: list[SpeechStarted | SpeechEnded], frame: AudioFrame, probability: float) -> list[SpeechStarted | SpeechEnded]:
        accepted: list[SpeechStarted | SpeechEnded] = []
        for event in events:
            match event:
                case SpeechStarted():
                    if self._suppress_self_trigger():
                        return []
                    if self._requires_barge_in_gate():
                        started, rejection = self.barge_in_gate.begin(event, frame, probability, self.detector.policy.threshold)
                        self._record_barge_in_rejection(rejection)
                        if started is None:
                            continue
                        self._count_barge_in_accepted()
                        await self.orchestrator.speech_started(started.start_frame_index)
                        accepted.append(started)
                        continue
                    await self.orchestrator.speech_started(event.start_frame_index)
                    accepted.append(event)
                case SpeechEnded():
                    if self.barge_in_gate.reject_short():
                        self._count_barge_in_rejection("short")
                        continue
                    accepted.append(event)
                case unreachable:
                    assert_never(unreachable)
        if self._requires_barge_in_gate() and self.barge_in_gate.pending is not None:
            started, rejection = self.barge_in_gate.advance(frame, probability, self.detector.policy.threshold)
            self._record_barge_in_rejection(rejection)
            if started is not None:
                self._count_barge_in_accepted()
                await self.orchestrator.speech_started(started.start_frame_index)
                accepted.append(started)
        return accepted

    def _requires_barge_in_gate(self) -> bool:
        return self.orchestrator.state in ACTIVE_BARGE_IN_STATES

    def _is_verified_nonspeech(self, probability: float) -> bool:
        return (
            not self._requires_barge_in_gate()
            and probability < self.detector.policy.threshold
            and self.detector.candidate_start_frame_index is None
            and self.detector.active_start_frame_index is None
            and self.detector.silence_start_frame_index is None
        )

    def _suppress_self_trigger(self) -> bool:
        if self.interruption_guard is None or not self.interruption_guard.should_suppress():
            return False
        self.diagnostics = LiveDiagnostics(
            self.diagnostics.capture_overruns, self.diagnostics.input_underruns,
            self.diagnostics.self_trigger_suppressions + 1, self.diagnostics.barge_in_accepted,
            self.diagnostics.barge_in_rejected_noise, self.diagnostics.barge_in_rejected_short,
        )
        return True

    def _count_capture_overrun(self) -> None:
        self.diagnostics = LiveDiagnostics(
            self.diagnostics.capture_overruns + 1, self.diagnostics.input_underruns,
            self.diagnostics.self_trigger_suppressions, self.diagnostics.barge_in_accepted,
            self.diagnostics.barge_in_rejected_noise, self.diagnostics.barge_in_rejected_short,
        )

    def _count_barge_in_accepted(self) -> None:
        self.diagnostics = LiveDiagnostics(
            self.diagnostics.capture_overruns, self.diagnostics.input_underruns,
            self.diagnostics.self_trigger_suppressions, self.diagnostics.barge_in_accepted + 1,
            self.diagnostics.barge_in_rejected_noise, self.diagnostics.barge_in_rejected_short,
            self.barge_in_gate.last_echo_correlation, self.diagnostics.barge_in_rejected_echo,
        )

    def _record_barge_in_rejection(self, rejection: GateRejection | None) -> None:
        match rejection:
            case "noise":
                self.diagnostics = LiveDiagnostics(
                    self.diagnostics.capture_overruns, self.diagnostics.input_underruns,
                    self.diagnostics.self_trigger_suppressions, self.diagnostics.barge_in_accepted,
                    self.diagnostics.barge_in_rejected_noise + 1, self.diagnostics.barge_in_rejected_short,
                )
            case "short":
                self._count_barge_in_rejection("short")
            case "echo":
                self.diagnostics = LiveDiagnostics(
                    self.diagnostics.capture_overruns, self.diagnostics.input_underruns,
                    self.diagnostics.self_trigger_suppressions, self.diagnostics.barge_in_accepted,
                    self.diagnostics.barge_in_rejected_noise, self.diagnostics.barge_in_rejected_short,
                    self.barge_in_gate.last_echo_correlation, self.diagnostics.barge_in_rejected_echo + 1,
                )
            case None:
                return

    def _count_barge_in_rejection(self, rejection: GateRejection) -> None:
        match rejection:
            case "noise":
                self._record_barge_in_rejection("noise")
            case "short":
                self.diagnostics = LiveDiagnostics(
                    self.diagnostics.capture_overruns, self.diagnostics.input_underruns,
                    self.diagnostics.self_trigger_suppressions, self.diagnostics.barge_in_accepted,
                    self.diagnostics.barge_in_rejected_noise, self.diagnostics.barge_in_rejected_short + 1,
                )
