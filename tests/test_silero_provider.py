import numpy as np
import torch

from holo_companion.audio.types import AudioFrame
from holo_companion.vad.base import VadPolicy
from holo_companion.vad.silero import SileroVadProvider, create_silero_vad_provider


class FakeModel:
    def __init__(self, probability: float = 0.75) -> None:
        self.probability = probability
        self.calls: list[tuple[torch.Tensor, int]] = []
        self.reset_count = 0

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.calls.append((samples, sample_rate))
        return torch.tensor([[self.probability]], dtype=torch.float32)

    def reset_states(self) -> None:
        self.reset_count += 1


def test_silero_provider_returns_probability_from_cpu_tensor() -> None:
    # Given
    model = FakeModel()
    provider = SileroVadProvider(model)
    audio = AudioFrame(samples=np.zeros((512,), dtype=np.float32), frame_index=7)

    # When
    probability = provider.process(audio)

    # Then
    assert probability == 0.75
    assert model.calls[0][0].device.type == "cpu"
    assert model.calls[0][0].shape == (1, 512)
    assert model.calls[0][1] == 16_000


def test_silero_provider_reset_delegates_to_model() -> None:
    # Given
    model = FakeModel()
    provider = SileroVadProvider(model)

    # When
    provider.reset()

    # Then
    assert model.reset_count == 1


def test_silero_factory_loads_packaged_onnx_model() -> None:
    # Given
    loaded: list[bool] = []
    model = FakeModel()

    def load_model(onnx: bool) -> FakeModel:
        loaded.append(onnx)
        return model

    # When
    provider = create_silero_vad_provider(
        VadPolicy(min_speech_ms=250, min_silence_ms=500, speech_pad_ms=30, threshold=0.5),
        load_model=load_model,
    )

    # Then
    assert loaded == [True]
    assert provider.process(AudioFrame(samples=np.zeros((512,), dtype=np.float32), frame_index=0)) == 0.75
