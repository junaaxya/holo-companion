import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from holo_companion.audio.capture import WaveWriter
from holo_companion.audio.types import AudioCliError
from holo_companion.tts.base import SynthesisRequest, SynthesisResult, SynthesisStatus, TTSAudioChunk, TTSProvider, TtsAudioFormat, TtsError, TtsErrorKind
from holo_companion.tts.elevenlabs import ElevenLabsConfig


class TtsConfigLoader(Protocol):
    def __call__(self) -> ElevenLabsConfig: ...


class TtsProviderFactory(Protocol):
    def __call__(self, config: ElevenLabsConfig) -> TTSProvider: ...


@dataclass(frozen=True, slots=True)
class TtsSmokeResult:
    output: Path
    audio_format: TtsAudioFormat
    provider: str
    model: str
    ttfa_seconds: float | None
    synthesis_seconds: float
    generated_seconds: float
    rtf: float
    received_bytes: int
    chunk_count: int


async def run_tts_smoke(
    request: SynthesisRequest,
    output: Path,
    config_loader: TtsConfigLoader,
    provider_factory: TtsProviderFactory,
    writer: WaveWriter,
) -> TtsSmokeResult:
    if output.exists():
        raise AudioCliError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provider = provider_factory(config_loader())
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.wav")
    chunks: list[TTSAudioChunk] = []
    terminal: SynthesisResult | None = None
    try:
        async for event in provider.stream(request):
            match event:
                case TTSAudioChunk():
                    chunks.append(event)
                case SynthesisResult():
                    terminal = event
        if terminal is None:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS stream ended without terminal result")
        if terminal.status is not SynthesisStatus.COMPLETED:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS synthesis was not completed")
        audio_format = validate_chunks(chunks)
        samples = np.ascontiguousarray(np.concatenate([chunk.samples for chunk in chunks]))
        writer.write_pcm16(temporary, samples, audio_format.sample_rate_hz)
        os.replace(temporary, output)
        metrics = terminal.metrics
        if metrics.provider is None or metrics.model is None:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS result missing provider metadata")
        return TtsSmokeResult(output, audio_format, metrics.provider, metrics.model, metrics.time_to_first_audio_seconds, metrics.synthesis_seconds, metrics.generated_audio_seconds, metrics.real_time_factor, metrics.received_byte_count, metrics.chunk_count)
    finally:
        if temporary.exists():
            temporary.unlink()
        await provider.aclose()


def validate_chunks(chunks: list[TTSAudioChunk]) -> TtsAudioFormat:
    if len(chunks) == 0:
        raise TtsError(TtsErrorKind.PROVIDER, "TTS stream returned no audio")
    audio_format = chunks[0].audio_format
    for sequence, chunk in enumerate(chunks):
        if chunk.sequence != sequence or chunk.audio_format != audio_format:
            raise TtsError(TtsErrorKind.PROVIDER, "TTS stream returned incompatible audio chunks")
    return audio_format


def format_tts_smoke_result(result: TtsSmokeResult) -> str:
    ttfa = "n/a" if result.ttfa_seconds is None else f"{result.ttfa_seconds:.3f} s"
    return (
        f"Provider: {result.provider}\nModel: {result.model}\nOutput: {result.output}\n"
        f"Sample rate: {result.audio_format.sample_rate_hz} Hz\nTTFA: {ttfa}\n"
        f"Total synthesis latency: {result.synthesis_seconds:.3f} s\n"
        f"Generated audio duration: {result.generated_seconds:.3f} s\nRTF: {result.rtf:.3f}\n"
        f"Bytes received: {result.received_bytes}\nChunk count: {result.chunk_count}\n"
    )
