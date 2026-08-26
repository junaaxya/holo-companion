import io
import json

import numpy as np

from holo_companion.audio.types import AudioFrame
from holo_companion.cli import run_cli
from holo_companion.vad.base import VadPolicy

from audio_fakes import FakeWaveWriter, StreamCall, fake_backend


class FakeVadProvider:
    def __init__(self) -> None:
        self.reset_count = 0

    def process(self, frame: AudioFrame) -> float:
        return 0.9 if frame.frame_index < 18 else 0.0

    def reset(self) -> None:
        self.reset_count += 1


def test_vad_test_command_emits_bounded_json_without_recording_artifact() -> None:
    # Given
    stdout = io.StringIO()
    backend = fake_backend()
    backend.stream_frames = [AudioFrame(samples=np.zeros((512,), dtype=np.float32), frame_index=index) for index in range(63)]
    provider = FakeVadProvider()

    def provider_factory(policy: VadPolicy) -> FakeVadProvider:
        assert policy == VadPolicy(min_speech_ms=500, min_silence_ms=700, speech_pad_ms=30, threshold=0.6)
        return provider

    # When
    code = run_cli(
        ["vad-test", "--seconds", "2"],
        backend,
        FakeWaveWriter(),
        stdout,
        io.StringIO(),
        vad_provider_factory=provider_factory,
    )

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert backend.record_calls == []
    assert backend.stream_calls == [StreamCall(16_000, 1, "float32", backend.default_input, 512)]
    assert payload["requested_input_device"] == "auto"
    assert payload["format"] == {"samplerate_hz": 16_000, "channels": 1, "dtype": "float32", "frame_samples": 512, "frame_ms": 32.0}
    assert payload["policy"] == {"threshold": 0.6, "min_speech_ms": 500, "min_silence_ms": 700, "speech_pad_ms": 30}
    assert payload["events"] == [
        {"type": "speech_started", "start_frame_index": 0, "detected_frame_index": 16, "start_time_ms": 0.0},
        {
            "type": "speech_ended",
            "start_frame_index": 0,
            "end_frame_index": 18,
            "detected_frame_index": 40,
            "duration_ms": 576.0,
            "end_latency_ms": 704.0,
        },
    ]
    assert payload["diagnostics"]["requested_frames"] == 63
    assert payload["diagnostics"]["processed_frames"] == 63
    assert payload["diagnostics"]["speech_started_count"] == 1
    assert payload["diagnostics"]["speech_ended_count"] == 1
    assert provider.reset_count == 1


def test_vad_test_command_rejects_invalid_seconds_before_capture() -> None:
    # Given
    stderr = io.StringIO()

    # When / Then
    try:
        run_cli(["vad-test", "--seconds", "0"], fake_backend(), FakeWaveWriter(), io.StringIO(), stderr)
    except SystemExit as error:
        assert error.code == 2
    assert "--seconds must be at least 0.1" in stderr.getvalue()


def test_vad_test_command_converts_runtime_audio_error_to_exit_2() -> None:
    # Given
    stderr = io.StringIO()
    backend = fake_backend(incompatible=True)

    # When
    code = run_cli(["vad-test", "--seconds", "1"], backend, FakeWaveWriter(), io.StringIO(), stderr)

    # Then
    assert code == 2
    assert "does not support" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
