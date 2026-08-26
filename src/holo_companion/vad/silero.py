from typing import Protocol

import torch

from holo_companion.audio.types import CAPTURE_SAMPLE_RATE, AudioFrame
from holo_companion.vad.base import VadPolicy


class SileroModel(Protocol):
    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor: ...

    def reset_states(self) -> None: ...


class SileroLoader(Protocol):
    def __call__(self, onnx: bool) -> SileroModel: ...


class SileroVadProvider:
    def __init__(self, model: SileroModel) -> None:
        self._model = model

    def process(self, frame: AudioFrame) -> float:
        samples = torch.from_numpy(frame.samples).to(device="cpu", dtype=torch.float32).unsqueeze(0)
        return float(self._model(samples, int(CAPTURE_SAMPLE_RATE)).item())

    def reset(self) -> None:
        self._model.reset_states()


def create_silero_vad_provider(
    policy: VadPolicy,
    load_model: SileroLoader | None = None,
) -> SileroVadProvider:
    del policy
    model_loader = _load_model if load_model is None else load_model
    return SileroVadProvider(model_loader(True))


def _load_model(onnx: bool) -> SileroModel:
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)
