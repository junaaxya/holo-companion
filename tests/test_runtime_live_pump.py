from dataclasses import dataclass, field

import anyio
import numpy as np
import pytest
import soundfile

from holo_companion.audio.types import AudioFrame, DeviceIndex
from holo_companion.audio.raw_capture import RawCaptureDumper
from holo_companion.runtime.live_pump import CaptureVadPump, InterruptionGuard, LiveDiagnostics
from holo_companion.runtime.barge_in import AdaptiveBargeInGate
from holo_companion.runtime.events import SpeechStarted
from holo_companion.runtime.metrics import TurnTimestamps
from holo_companion.runtime.state import RuntimeState
from holo_companion.runtime.turn import TurnResult
from holo_companion.stt.base import Utterance
from holo_companion.stt.collector import UtteranceCollector
from holo_companion.vad.base import TurnDetector, VadPolicy


def frame(index: int) -> AudioFrame:
    return AudioFrame(samples=np.full((512,), index / 10.0, dtype=np.float32), frame_index=index)


def level_frame(index: int, level: float) -> AudioFrame:
    return AudioFrame(samples=np.full((512,), level, dtype=np.float32), frame_index=index)


def waveform_frame(index: int, samples: np.ndarray) -> AudioFrame:
    return AudioFrame(samples=np.ascontiguousarray(samples, dtype=np.float32), frame_index=index)


@dataclass(slots=True)
class FakeBackend:
    frames: tuple[AudioFrame, ...]
    requested: tuple[int, int, str, int, int] | None = None

    def frame_stream(self, samplerate: int, channels: int, dtype: str, device: DeviceIndex, blocksize: int):
        self.requested = (samplerate, channels, dtype, int(device), blocksize)
        yield from self.frames


@dataclass(slots=True)
class ProbabilityVad:
    probabilities: tuple[float, ...]
    index: int = 0
    reset_count: int = 0

    def process(self, audio_frame: AudioFrame) -> float:
        del audio_frame
        probability = self.probabilities[self.index]
        self.index += 1
        return probability

    def reset(self) -> None:
        self.reset_count += 1


@dataclass(slots=True)
class RecordingOrchestrator:
    started: int = 0
    utterances: list[Utterance] = field(default_factory=list)
    state: RuntimeState = RuntimeState.IDLE

    async def speech_started(self, start_frame_index: int | None = None) -> None:
        del start_frame_index
        self.started += 1

    async def speech_ended(self, utterance: Utterance) -> TurnResult:
        self.utterances.append(utterance)
        return TurnResult(1, 1, "halo", "", TurnTimestamps())


@dataclass(slots=True)
class BlockingOrchestrator:
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event = field(default_factory=anyio.Event)
    started: int = 0
    state: RuntimeState = RuntimeState.IDLE

    async def speech_started(self, start_frame_index: int | None = None) -> None:
        del start_frame_index
        self.started += 1

    async def speech_ended(self, utterance: Utterance) -> TurnResult:
        del utterance
        self.state = RuntimeState.SPEAKING
        self.entered.set()
        await self.release.wait()
        return TurnResult(1, 1, None, "", TurnTimestamps())


@dataclass(slots=True)
class InterruptingOrchestrator:
    entered: anyio.Event = field(default_factory=anyio.Event)
    release: anyio.Event = field(default_factory=anyio.Event)
    started: int = 0
    stopped: int = 0
    state: RuntimeState = RuntimeState.IDLE

    async def speech_started(self, start_frame_index: int | None = None) -> None:
        del start_frame_index
        self.started += 1
        if self.entered.is_set():
            self.stopped += 1

    async def speech_ended(self, utterance: Utterance) -> TurnResult:
        del utterance
        self.state = RuntimeState.SPEAKING
        self.entered.set()
        await self.release.wait()
        return TurnResult(1, 1, None, "", TurnTimestamps())


@dataclass(slots=True)
class SpeakingOrchestrator:
    started: int = 0
    state: RuntimeState = RuntimeState.SPEAKING

    async def speech_started(self, start_frame_index: int | None = None) -> None:
        del start_frame_index
        self.started += 1

    async def speech_ended(self, utterance: Utterance) -> TurnResult:
        del utterance
        return TurnResult(1, 1, None, "", TurnTimestamps())


def detector() -> TurnDetector:
    return TurnDetector(VadPolicy(min_speech_ms=64, min_silence_ms=64, speech_pad_ms=30, threshold=0.5))


@pytest.mark.anyio
async def test_pump_uses_canonical_capture_and_emits_valid_utterance() -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(6)))
    vad = ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0))
    orchestrator = RecordingOrchestrator()
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(7))

    # When
    diagnostics = await pump.run()

    # Then
    assert backend.requested == (16_000, 1, "float32", 7, 512)
    assert orchestrator.started == 1
    assert len(orchestrator.utterances) == 1
    assert orchestrator.utterances[0].start_frame_index == 0
    assert orchestrator.utterances[0].end_frame_index == 3
    assert diagnostics == LiveDiagnostics(capture_overruns=0, input_underruns=0, self_trigger_suppressions=0)


@pytest.mark.anyio
async def test_pump_reports_completed_turn_without_provider_or_hardware() -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(6)))
    vad = ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0))
    orchestrator = RecordingOrchestrator()
    results: list[TurnResult] = []

    async def record(result: TurnResult) -> None:
        results.append(result)

    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1), on_turn_complete=record)

    # When
    await pump.run()

    # Then
    assert [result.transcript for result in results] == ["halo"]


@pytest.mark.anyio
async def test_pump_optional_dump_writes_exact_canonical_utterance(tmp_path) -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(6)))
    vad = ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0))
    orchestrator = RecordingOrchestrator()
    dumped: list[Utterance] = []

    def dump(utterance: Utterance) -> None:
        dumped.append(utterance)
        soundfile.write(tmp_path / "turn-0001.wav", utterance.samples, utterance.sample_rate_hz, format="WAV", subtype="PCM_16")

    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1), dump_utterance=dump)

    # When
    await pump.run()

    # Then
    samples, sample_rate = soundfile.read(tmp_path / "turn-0001.wav", dtype="float32")
    assert len(dumped) == 1
    assert sample_rate == 16_000
    assert samples.ndim == 1
    np.testing.assert_allclose(samples, dumped[0].samples, atol=1 / 32_768)


@pytest.mark.anyio
async def test_pump_without_dump_writes_no_artifact(tmp_path) -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(6)))
    vad = ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0))
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), RecordingOrchestrator(), DeviceIndex(1))

    # When
    await pump.run()

    # Then
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_raw_capture_dump_preserves_every_pre_vad_frame_in_order(tmp_path) -> None:
    # Given
    frames = tuple(frame(index) for index in range(6))
    dump_path = tmp_path / "raw-live.wav"
    pump = CaptureVadPump(
        FakeBackend(frames),
        ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0)),
        detector(),
        UtteranceCollector(history_frames=8),
        RecordingOrchestrator(),
        DeviceIndex(1),
        raw_capture_dumper=RawCaptureDumper(dump_path),
    )

    # When
    await pump.run()

    # Then
    samples, sample_rate = soundfile.read(dump_path, dtype="float32")
    assert sample_rate == 16_000
    assert samples.ndim == 1
    np.testing.assert_allclose(samples, np.concatenate([audio_frame.samples for audio_frame in frames]), atol=1 / 32_768)


@pytest.mark.anyio
async def test_pump_bounded_capture_buffer_counts_overrun_without_unbounded_growth() -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(20)))
    vad = ProbabilityVad(tuple(0.9 if index < 3 else 0.0 for index in range(20)))
    orchestrator = BlockingOrchestrator()
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1), frame_capacity=0)

    # When / Then
    with anyio.move_on_after(0.5) as timeout:
        async with anyio.create_task_group() as tg:
            tg.start_soon(pump.run)
            await orchestrator.entered.wait()
            while pump.diagnostics.capture_overruns == 0:
                await anyio.sleep(0.01)
            assert pump.diagnostics.capture_overruns > 0
            orchestrator.release.set()
            tg.cancel_scope.cancel()
    assert not timeout.cancelled_caught


@pytest.mark.anyio
async def test_pump_keeps_vad_alive_while_assistant_turn_runs_and_rejects_short_barge_in() -> None:
    # Given
    backend = FakeBackend(tuple(frame(index) for index in range(12)))
    vad = ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0))
    orchestrator = InterruptingOrchestrator()
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1))

    # When
    async with anyio.create_task_group() as tg:
        tg.start_soon(pump.run)
        await orchestrator.entered.wait()
        while vad.index < 12:
            await anyio.lowlevel.checkpoint()
        orchestrator.release.set()

    # Then
    assert vad.index == 12
    assert orchestrator.started == 1
    assert orchestrator.stopped == 0
    assert pump.diagnostics.barge_in_rejected_short == 1
    assert pump.diagnostics.capture_overruns == 0


@pytest.mark.anyio
async def test_guard_suppresses_initial_playback_window_then_allows_later_interrupt() -> None:
    # Given
    now = 0.0
    guard = InterruptionGuard(guard_ms=300, monotonic_seconds=lambda: now)
    backend = FakeBackend(tuple(frame(index) for index in range(10)))
    vad = ProbabilityVad((0.9,) * 10)
    orchestrator = RecordingOrchestrator()
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1), interruption_guard=guard)
    guard.mark_playback_consumed()

    # When
    now = 0.2
    await pump.run()

    # Then
    assert orchestrator.started == 0
    assert pump.diagnostics.self_trigger_suppressions == 1

    # Given
    vad = ProbabilityVad((0.9,) * 10)
    orchestrator = RecordingOrchestrator()
    pump = CaptureVadPump(backend, vad, detector(), UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1), interruption_guard=guard)

    # When
    now = 0.31
    await pump.run()

    # Then
    assert orchestrator.started == 1


@pytest.mark.anyio
@pytest.mark.parametrize("state", [RuntimeState.TRANSCRIBING, RuntimeState.THINKING, RuntimeState.SPEAKING])
async def test_active_turn_gate_rejects_low_level_background_noise(state: RuntimeState) -> None:
    # Given
    frames = tuple(level_frame(index, 0.01) for index in range(4)) + tuple(level_frame(index, 0.02) for index in range(4, 15)) + tuple(level_frame(index, 0.01) for index in range(15, 18))
    orchestrator = SpeakingOrchestrator(state=state)
    pump = CaptureVadPump(
        FakeBackend(frames), ProbabilityVad((0.0,) * 4 + (0.9,) * 11 + (0.0,) * 3), detector(),
        UtteranceCollector(history_frames=16), orchestrator, DeviceIndex(1),
        barge_in_gate=AdaptiveBargeInGate(noise_floor=0.01),
    )

    # When
    await pump.run()

    # Then
    assert orchestrator.started == 0
    assert pump.diagnostics.barge_in_rejected_noise == 1
    assert pump.diagnostics.barge_in_accepted == 0


@pytest.mark.anyio
async def test_speaking_gate_rejects_short_close_mic_transient() -> None:
    # Given
    frames = tuple(level_frame(index, 0.01) for index in range(4)) + tuple(level_frame(index, 0.30) for index in range(4, 7)) + tuple(level_frame(index, 0.01) for index in range(7, 10))
    orchestrator = SpeakingOrchestrator()
    pump = CaptureVadPump(
        FakeBackend(frames), ProbabilityVad((0.0,) * 4 + (0.9,) * 3 + (0.0,) * 3), detector(),
        UtteranceCollector(history_frames=16), orchestrator, DeviceIndex(1),
    )

    # When
    await pump.run()

    # Then
    assert orchestrator.started == 0
    assert pump.diagnostics.barge_in_rejected_short == 1
    assert pump.diagnostics.barge_in_accepted == 0


@pytest.mark.anyio
async def test_speaking_gate_accepts_sustained_close_mic_speech() -> None:
    # Given
    frames = tuple(level_frame(index, 0.01) for index in range(4)) + tuple(level_frame(index, 0.30) for index in range(4, 15)) + tuple(level_frame(index, 0.01) for index in range(15, 18))
    orchestrator = SpeakingOrchestrator()
    pump = CaptureVadPump(
        FakeBackend(frames), ProbabilityVad((0.0,) * 4 + (0.9,) * 11 + (0.0,) * 3), detector(),
        UtteranceCollector(history_frames=16), orchestrator, DeviceIndex(1),
        barge_in_gate=AdaptiveBargeInGate(noise_floor=0.01),
    )

    # When
    await pump.run()

    # Then
    assert orchestrator.started == 1
    assert pump.diagnostics.barge_in_accepted == 1
    assert pump.diagnostics.barge_in_rejected_noise == 0
    assert pump.diagnostics.barge_in_rejected_short == 0


@pytest.mark.anyio
async def test_listening_keeps_normal_vad_behavior_for_low_level_speech() -> None:
    # Given
    frames = tuple(level_frame(index, 0.01) for index in range(3)) + tuple(level_frame(index, 0.02) for index in range(3, 6))
    orchestrator = RecordingOrchestrator()
    pump = CaptureVadPump(
        FakeBackend(frames), ProbabilityVad((0.9, 0.9, 0.9, 0.0, 0.0, 0.0)), detector(),
        UtteranceCollector(history_frames=8), orchestrator, DeviceIndex(1),
    )

    # When
    await pump.run()

    # Then
    assert orchestrator.started == 1
    assert len(orchestrator.utterances) == 1
    assert pump.diagnostics.barge_in_accepted == 0
    assert pump.diagnostics.barge_in_rejected_noise == 0
    assert pump.diagnostics.barge_in_rejected_short == 0


@pytest.mark.anyio
async def test_listening_user_speech_does_not_raise_ambient_noise_floor() -> None:
    # Given
    frames = tuple(level_frame(index, 0.01) for index in range(3)) + tuple(level_frame(index, 0.30) for index in range(3, 6)) + tuple(level_frame(index, 0.01) for index in range(6, 9))
    gate = AdaptiveBargeInGate()
    pump = CaptureVadPump(
        FakeBackend(frames), ProbabilityVad((0.0,) * 3 + (0.9,) * 3 + (0.0,) * 3), detector(),
        UtteranceCollector(history_frames=8), RecordingOrchestrator(), DeviceIndex(1), barge_in_gate=gate,
    )

    # When
    await pump.run()

    # Then
    assert gate.noise_floor == pytest.approx(0.01)
    assert tuple(gate.ambient_history) == pytest.approx((0.01, 0.01, 0.01))


def test_barge_in_gate_rejects_direct_playback_leak() -> None:
    # Given
    reference = np.sin(np.linspace(0.0, 12.0, 5_632, dtype=np.float32)).astype(np.float32)
    gate = AdaptiveBargeInGate(noise_floor=0.01)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, reference[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, reference[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is None
    assert rejection == "echo"
    assert gate.last_echo_correlation is not None and gate.last_echo_correlation > 0.85


def test_barge_in_gate_rejects_delayed_attenuated_echo() -> None:
    # Given
    reference = np.cos(np.linspace(0.0, 12.0, 6_144, dtype=np.float32)).astype(np.float32)
    delayed = reference[512:6_144] * 0.4
    gate = AdaptiveBargeInGate(noise_floor=0.01)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, delayed[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, delayed[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is None
    assert rejection == "echo"


def test_barge_in_gate_rejects_room_colored_delayed_echo() -> None:
    # Given
    reference = np.sin(np.linspace(0.0, 32.0, 7_168, dtype=np.float32)).astype(np.float32)
    echo = np.convolve(reference, np.asarray([0.7, 0.2, 0.1], dtype=np.float32), mode="same")[1_024:6_656] * 0.35
    gate = AdaptiveBargeInGate(noise_floor=0.01)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, echo[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, echo[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is None
    assert rejection == "echo"
    assert gate.last_echo_best_delay_ms is not None and gate.last_echo_best_delay_ms >= 0.0


def test_barge_in_gate_accepts_unrelated_close_mic_speech() -> None:
    # Given
    reference = np.sin(np.linspace(0.0, 12.0, 5_632, dtype=np.float32)).astype(np.float32)
    speech = np.sign(np.sin(np.linspace(0.0, 90.0, 5_632, dtype=np.float32))).astype(np.float32) * 0.3
    gate = AdaptiveBargeInGate(noise_floor=0.01)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, speech[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, speech[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is not None
    assert rejection is None
    assert gate.last_echo_correlation is not None and gate.last_echo_correlation < 0.85


def test_barge_in_gate_accepts_speech_with_weaker_echo() -> None:
    # Given
    reference = np.sin(np.linspace(0.0, 12.0, 5_632, dtype=np.float32)).astype(np.float32)
    speech = np.sign(np.sin(np.linspace(0.0, 90.0, 5_632, dtype=np.float32))).astype(np.float32) * 0.3
    mixed = speech + reference * 0.05
    gate = AdaptiveBargeInGate(noise_floor=0.01)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, mixed[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, mixed[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is not None
    assert rejection is None
    assert gate.last_echo_best_delay_ms is not None
    assert gate.last_playback_reference_rms is not None


def test_barge_in_gate_accepts_normal_volume_speech_with_frozen_ambient_floor() -> None:
    # Given
    reference = np.sin(np.linspace(0.0, 12.0, 5_632, dtype=np.float32)).astype(np.float32)
    speech = np.sign(np.sin(np.linspace(0.0, 90.0, 5_632, dtype=np.float32))).astype(np.float32) * 0.12
    gate = AdaptiveBargeInGate(noise_floor=0.02)
    gate.observe_playback(reference, 16_000)
    event = SpeechStarted(start_frame_index=0, detected_frame_index=2, start_time_ms=0.0)

    # When
    started, rejection = gate.begin(event, waveform_frame(2, speech[:512]), 0.9, 0.5)
    for index in range(3, 11):
        started, rejection = gate.advance(waveform_frame(index, speech[index * 512 : (index + 1) * 512]), 0.9, 0.5)

    # Then
    assert started is not None
    assert rejection is None
    assert gate.last_rms_ratio is not None and gate.last_rms_ratio > 3.0
    assert gate.last_reject_reason is None
