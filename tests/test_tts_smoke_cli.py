from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import soundfile

from holo_companion.audio.capture import SoundFileWaveWriter
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
)
from holo_companion.tts.elevenlabs import ELEVENLABS_V3_CONVERSATIONAL_MODEL, ElevenLabsConfig


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeProvider:
    events: tuple[TTSAudioChunk | SynthesisResult, ...]
    error: TtsError | None = None
    request: SynthesisRequest | None = None
    closed: bool = False

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[TTSAudioChunk | SynthesisResult]:
        self.request = request
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


def completed_events() -> tuple[TTSAudioChunk | SynthesisResult, ...]:
    audio_format = TtsAudioFormat(sample_rate_hz=24_000)
    chunks = (
        TTSAudioChunk.from_samples(np.ascontiguousarray(np.array([0.0, 0.5], dtype=np.float32)), audio_format, 0, 0.0),
        TTSAudioChunk.from_samples(np.ascontiguousarray(np.array([-0.5, 0.0], dtype=np.float32)), audio_format, 1, 2 / 24_000),
    )
    return (*chunks, SynthesisResult(
        request=SynthesisRequest(text="unused"),
        status=SynthesisStatus.COMPLETED,
        metrics=SynthesisMetrics(
            request_started_at_seconds=1.0,
            time_to_first_audio_seconds=0.2,
            synthesis_seconds=0.5,
            generated_audio_seconds=4 / 24_000,
            real_time_factor=3_000.0,
            chunk_count=2,
            received_byte_count=8,
            provider="elevenlabs",
            model=ELEVENLABS_V3_CONVERSATIONAL_MODEL,
        ),
    ))


def config_loader() -> ElevenLabsConfig:
    return ElevenLabsConfig(api_key="test-secret", voice_id="voice-a", model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL)


def run_smoke(
    argv: list[str],
    provider: FakeProvider,
    config: Callable[[], ElevenLabsConfig] = config_loader,
) -> tuple[int, str, str]:
    from holo_companion.cli import run_cli

    stdout = StringIO()
    stderr = StringIO()
    exit_code = run_cli(
        argv,
        stdout=stdout,
        stderr=stderr,
        tts_config_loader=config,
        tts_provider_factory=lambda _config: provider,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_tts_smoke_writes_ordered_pcm16_wav_and_metrics(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "recordings" / "tts-eval" / "holo.wav"
    provider = FakeProvider(completed_events())

    # When
    exit_code, stdout, stderr = run_smoke([
        "tts-smoke", "--text", "Ih... ternyata kamu belum tidur juga.", "--emotion", "playful", "--intensity", "0.6", "--output", str(output),
    ], provider)

    # Then
    assert exit_code == 0
    assert stderr == ""
    assert provider.request == SynthesisRequest(text="Ih... ternyata kamu belum tidur juga.", style=StyleHints(emotion="playful", intensity=0.6))
    assert provider.closed
    samples, sample_rate = soundfile.read(output, dtype="float32")
    assert sample_rate == 24_000
    assert samples.ndim == 1
    np.testing.assert_allclose(samples, np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32), atol=1 / 32_768)
    assert "Provider: elevenlabs" in stdout
    assert "Model: eleven_v3_conversational" in stdout
    assert f"Output: {output}" in stdout
    assert "Sample rate: 24000 Hz" in stdout
    assert "TTFA: 0.200 s" in stdout
    assert "Total synthesis latency: 0.500 s" in stdout
    assert "Generated audio duration: 0.000 s" in stdout
    assert "RTF: 3000.000" in stdout
    assert "Bytes received: 8" in stdout
    assert "Chunk count: 2" in stdout
    assert "test-secret" not in stdout


def test_tts_smoke_rejects_missing_config_without_provider_call(tmp_path: Path) -> None:
    # Given
    provider = FakeProvider(completed_events())

    def missing_config() -> ElevenLabsConfig:
        raise TtsError(TtsErrorKind.CONFIG, "ELEVENLABS_API_KEY is not configured")

    # When
    exit_code, stdout, stderr = run_smoke(["tts-smoke", "--text", "Halo", "--output", str(tmp_path / "out.wav")], provider, missing_config)

    # Then
    assert exit_code == 2
    assert stdout == ""
    assert "ELEVENLABS_API_KEY is not configured" in stderr
    assert provider.request is None
    assert not provider.closed


@pytest.mark.parametrize(
    ("events", "error"),
    [
        ((), None),
        (completed_events()[:-1], None),
        ((), TtsError(TtsErrorKind.PROVIDER, "authentication failed for test-secret")),
    ],
)
def test_tts_smoke_removes_output_on_empty_or_provider_failure(
    tmp_path: Path,
    events: tuple[TTSAudioChunk | SynthesisResult, ...],
    error: TtsError | None,
) -> None:
    # Given
    output = tmp_path / "nested" / "out.wav"
    provider = FakeProvider(events=events, error=error)

    # When
    exit_code, stdout, stderr = run_smoke(["tts-smoke", "--text", "Halo", "--output", str(output)], provider)

    # Then
    assert exit_code == 2
    assert stdout == ""
    assert "test-secret" not in stderr
    assert not output.exists()
    assert provider.closed


def test_tts_smoke_rejects_existing_output_without_synthesis(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "out.wav"
    SoundFileWaveWriter().write_pcm16(output, np.ascontiguousarray(np.array([0.0], dtype=np.float32)), 24_000)
    provider = FakeProvider(completed_events())

    # When
    exit_code, stdout, stderr = run_smoke(["tts-smoke", "--text", "Halo", "--output", str(output)], provider)

    # Then
    assert exit_code == 2
    assert stdout == ""
    assert "output exists" in stderr
    assert provider.request is None


def test_tts_smoke_prints_allowlisted_provider_diagnostic_without_secret(tmp_path: Path) -> None:
    # Given
    secret = "test-secret"
    diagnostic = TtsProviderDiagnostic(
        stage="websocket_handshake",
        status_code=401,
        error="transport_error",
        error_class="handshake",
        message="Unauthorized [redacted]",
        model=ELEVENLABS_V3_CONVERSATIONAL_MODEL,
    )
    provider = FakeProvider((), TtsError(TtsErrorKind.PROVIDER, f"failure {secret}", diagnostic))

    # When
    exit_code, stdout, stderr = run_smoke(["tts-smoke", "--text", "Halo", "--output", str(tmp_path / "out.wav")], provider)

    # Then
    assert exit_code == 2
    assert stdout == ""
    assert "stage: websocket_handshake" in stderr
    assert "status_code: 401" in stderr
    assert "error: transport_error" in stderr
    assert "error_class: handshake" in stderr
    assert "message: Unauthorized [redacted]" in stderr
    assert f"model: {ELEVENLABS_V3_CONVERSATIONAL_MODEL}" in stderr
    assert secret not in stderr
