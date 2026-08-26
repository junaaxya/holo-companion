import io
import json
from pathlib import Path

import numpy as np
import pytest

from holo_companion.audio.types import AudioFrame
from holo_companion.cli import run_cli
from holo_companion.stt.base import SttConfig, SttError, Transcript, TranscriptSegment, Utterance
from holo_companion.stt.benchmark import score_transcript
from holo_companion.stt.io import load_canonical_wav
from holo_companion.vad.base import VadPolicy

from audio_fakes import FakeWaveWriter, fake_backend


class FakeSttProvider:
    def transcribe(self, utterance: Utterance, language_hint: str | None = None) -> Transcript:
        return Transcript(
            text=" Halo  Dunia",
            language="id" if language_hint is None else language_hint,
            language_probability=0.9,
            audio_duration_seconds=utterance.duration_seconds,
            processing_seconds=0.5,
            real_time_factor=0.5 / utterance.duration_seconds,
            model_name="large-v3-turbo",
            device="cpu",
            compute_type="int8",
            segments=(TranscriptSegment(0.0, utterance.duration_seconds, " Halo  Dunia"),),
        )


def fake_stt_factory(config: SttConfig) -> FakeSttProvider:
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    return FakeSttProvider()


def test_load_canonical_wav_rejects_non_16000_mono(tmp_path: Path) -> None:
    # Given
    wav = tmp_path / "bad.wav"
    import soundfile

    soundfile.write(wav, np.zeros((16, 2), dtype=np.float32), 8_000)

    # When / Then
    with pytest.raises(SttError, match="mono 16000 Hz"):
        load_canonical_wav(wav)


def test_stt_wav_outputs_stable_json_for_repeated_inputs(tmp_path: Path) -> None:
    # Given
    wav = tmp_path / "sample.wav"
    import soundfile

    soundfile.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    stdout = io.StringIO()

    # When
    code = run_cli(["stt-wav", "--input", str(wav), "--input", str(wav)], fake_backend(), FakeWaveWriter(), stdout, io.StringIO(), stt_provider_factory=fake_stt_factory)

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert [entry["audio_path"] for entry in payload["results"]] == [str(wav), str(wav)]
    assert payload["results"][0]["transcript"]["text"] == " Halo  Dunia"


def test_stt_wav_converts_provider_error_to_exit_2(tmp_path: Path) -> None:
    # Given
    wav = tmp_path / "sample.wav"
    import soundfile

    soundfile.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)

    def failing_factory(config: SttConfig) -> FakeSttProvider:
        raise SttError("load failed")

    stderr = io.StringIO()

    # When
    code = run_cli(["stt-wav", "--input", str(wav)], fake_backend(), FakeWaveWriter(), io.StringIO(), stderr, stt_provider_factory=failing_factory)

    # Then
    assert code == 2
    assert "load failed" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_benchmark_scores_normalized_text_without_changing_raw_output() -> None:
    # Given / When
    score = score_transcript(" Halo  Dunia", "halo dunia")

    # Then
    assert score.exact is True
    assert score.word_error_rate == 0.0
    assert score.char_error_rate == 0.0


def test_stt_benchmark_outputs_per_sample_and_aggregate_json(tmp_path: Path) -> None:
    # Given
    wav = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.json"
    import soundfile

    soundfile.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    manifest.write_text(json.dumps({"entries": [{"audio_path": "sample.wav", "expected_text": "halo dunia", "language": "id"}]}), encoding="utf-8")
    stdout = io.StringIO()

    # When
    code = run_cli(["stt-benchmark", "--manifest", str(manifest)], fake_backend(), FakeWaveWriter(), stdout, io.StringIO(), stt_provider_factory=fake_stt_factory)

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["manifest_path"] == str(manifest)
    assert payload["results"][0]["score"]["exact"] is True
    assert payload["aggregate"]["exact_rate"] == 1.0


def test_vad_test_transcribe_uses_single_stream_and_no_record() -> None:
    # Given
    stdout = io.StringIO()
    backend = fake_backend()
    backend.stream_frames = [AudioFrame(samples=np.zeros((512,), dtype=np.float32), frame_index=index) for index in range(63)]

    class FakeVadProvider:
        def process(self, frame: AudioFrame) -> float:
            return 0.9 if frame.frame_index < 18 else 0.0

        def reset(self) -> None:
            return None

    def vad_factory(policy: VadPolicy) -> FakeVadProvider:
        return FakeVadProvider()

    # When
    code = run_cli(["vad-test", "--seconds", "2", "--transcribe"], backend, FakeWaveWriter(), stdout, io.StringIO(), vad_provider_factory=vad_factory, stt_provider_factory=fake_stt_factory)

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert len(backend.stream_calls) == 1
    assert backend.record_calls == []
    assert payload["transcripts"][0]["text"] == " Halo  Dunia"
