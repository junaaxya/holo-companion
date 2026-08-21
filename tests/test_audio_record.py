import io
import json
from pathlib import Path

import numpy as np
import pytest

from holo_companion.audio.capture import record_wav
from holo_companion.audio.types import CAPTURE_DTYPE, CAPTURE_SAMPLE_RATE, AudioCliError, CaptureRequest, DeviceIndex, Frames, Seconds
from holo_companion.cli import run_cli

from audio_fakes import CheckCall, FakeWaveWriter, RecordCall, fake_backend


def test_record_wav_checks_records_and_writes_exact_contract_when_auto_selected(tmp_path: Path) -> None:
    # Given
    samples = np.array([[0.0], [0.5], [-1.0], [1.0]], dtype=np.float32)
    backend = fake_backend(samples=samples)
    writer = FakeWaveWriter()
    output = tmp_path / "mic-test.wav"

    # When
    result = record_wav(
        CaptureRequest(seconds=Seconds(0.25), output_path=output, requested_input_device="auto", overwrite=False),
        backend,
        writer,
    )

    # Then
    assert backend.check_calls == [CheckCall(DeviceIndex(1), 16_000, 1, CAPTURE_DTYPE)]
    assert backend.record_calls == [RecordCall(Frames(4_000), 16_000, 1, CAPTURE_DTYPE, DeviceIndex(1))]
    assert writer.writes == [(output, samples, int(CAPTURE_SAMPLE_RATE))]
    assert result.format.wav_subtype == "PCM_16"
    assert result.diagnostics.clipping_samples == 2


def test_record_wav_fails_when_output_exists_without_overwrite(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "existing.wav"
    output.write_bytes(b"old")

    # When / Then
    with pytest.raises(AudioCliError, match="output exists"):
        record_wav(
            CaptureRequest(seconds=Seconds(1.0), output_path=output, requested_input_device="auto", overwrite=False),
            fake_backend(),
            FakeWaveWriter(),
        )


def test_record_wav_fails_when_device_rejects_capture_settings(tmp_path: Path) -> None:
    # Given
    backend = fake_backend(incompatible=True)

    # When / Then
    with pytest.raises(AudioCliError, match="does not support"):
        record_wav(
            CaptureRequest(
                seconds=Seconds(1.0),
                output_path=tmp_path / "mic-test.wav",
                requested_input_device="auto",
                overwrite=False,
            ),
            backend,
            FakeWaveWriter(),
        )


def test_record_command_emits_json_when_default_input_is_used(tmp_path: Path) -> None:
    # Given
    stdout = io.StringIO()

    # When
    code = run_cli(
        ["record", "--seconds", "0.25", "--output", str(tmp_path / "mic-test.wav")],
        fake_backend(),
        FakeWaveWriter(),
        stdout,
        io.StringIO(),
    )

    # Then
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["requested_input_device"] == "auto"
    assert payload["format"]["samplerate_hz"] == 16_000
    assert payload["format"]["dtype"] == "float32"
    assert payload["diagnostics"]["frames"] == 4


def test_record_command_rejects_numeric_input_device_id(tmp_path: Path) -> None:
    # Given
    stderr = io.StringIO()

    # When
    code = run_cli(
        ["record", "--seconds", "1", "--output", str(tmp_path / "mic-test.wav"), "--input-device", "1"],
        fake_backend(),
        FakeWaveWriter(),
        io.StringIO(),
        stderr,
    )

    # Then
    assert code == 2
    assert "not a numeric device id" in stderr.getvalue()


def test_record_command_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "existing.wav"
    output.write_bytes(b"old")
    stderr = io.StringIO()

    # When
    code = run_cli(
        ["record", "--seconds", "1", "--output", str(output)],
        fake_backend(),
        FakeWaveWriter(),
        io.StringIO(),
        stderr,
    )

    # Then
    assert code == 2
    assert "output exists" in stderr.getvalue()


def test_record_command_rejects_invalid_duration_before_capture(tmp_path: Path) -> None:
    # Given
    stderr = io.StringIO()

    # When / Then
    with pytest.raises(SystemExit) as exit_info:
        run_cli(
            ["record", "--seconds", "0", "--output", str(tmp_path / "mic-test.wav")],
            fake_backend(),
            FakeWaveWriter(),
            io.StringIO(),
            stderr,
        )
    assert exit_info.value.code == 2
