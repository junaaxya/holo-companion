from pathlib import Path

import numpy as np
import pytest

from holo_companion.stt.base import SttConfig, SttError, Utterance


def test_stt_config_defaults_to_cpu_int8_external_cache() -> None:
    # Given / When
    config = SttConfig()

    # Then
    assert config.model_name == "large-v3-turbo"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.cache_dir == Path.home() / ".cache" / "holo-companion" / "models"
    assert config.local_files_only is False
    assert config.cpu_threads == 0
    assert config.num_workers == 1


def test_stt_config_rejects_unsupported_model() -> None:
    # Given / When / Then
    with pytest.raises(SttError, match="unsupported STT model"):
        SttConfig(model_name="medium")


def test_utterance_accepts_canonical_contiguous_float32() -> None:
    # Given
    samples = np.ascontiguousarray(np.array([0.0, 0.25], dtype=np.float32))

    # When
    utterance = Utterance.from_samples(samples)

    # Then
    assert utterance.samples is samples
    assert utterance.duration_seconds == 0.000125


def test_utterance_rejects_non_finite_audio() -> None:
    # Given
    samples = np.array([0.0, np.nan], dtype=np.float32)

    # When / Then
    with pytest.raises(SttError, match="finite"):
        Utterance.from_samples(samples)


def test_utterance_rejects_non_contiguous_audio() -> None:
    # Given
    samples = np.zeros((2, 2), dtype=np.float32)[:, 0]

    # When / Then
    with pytest.raises(SttError, match="contiguous"):
        Utterance.from_samples(samples)
