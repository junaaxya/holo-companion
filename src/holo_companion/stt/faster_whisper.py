from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from numpy.typing import NDArray
import numpy as np

from holo_companion.stt.base import SttConfig, SttError, Transcript, TranscriptSegment, Utterance


class WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class WhisperInfo(Protocol):
    language: str
    language_probability: float


class WhisperModelLike(Protocol):
    def transcribe(self, samples: NDArray[np.float32], language: str | None, task: str, vad_filter: bool) -> tuple[Iterable[WhisperSegment], WhisperInfo]: ...


WhisperModelLoader = Callable[[str, str, str, str, bool, int, int], WhisperModelLike]
Clock = Callable[[], float]


def load_whisper_model(model_name: str, device: str, compute_type: str, download_root: str, local_files_only: bool, cpu_threads: int, num_workers: int) -> WhisperModelLike:
    try:
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as error:
        raise SttError("failed to import Faster-Whisper") from error
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        local_files_only=local_files_only,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


@dataclass(frozen=True, slots=True)
class FasterWhisperSttProvider:
    config: SttConfig
    loader: WhisperModelLoader = load_whisper_model
    monotonic_seconds: Clock = monotonic

    def transcribe(self, utterance: Utterance, language_hint: str | None = None) -> Transcript:
        try:
            model = self.loader(
                self.config.model_name,
                self.config.device,
                self.config.compute_type,
                str(self.config.cache_dir),
                self.config.local_files_only,
                self.config.cpu_threads,
                self.config.num_workers,
            )
        except (ValueError, RuntimeError, OSError) as error:
            raise SttError("failed to load Faster-Whisper model") from error
        start_seconds = self.monotonic_seconds()
        try:
            language = None if language_hint == "auto" else language_hint
            lazy_segments, info = model.transcribe(utterance.samples, language=language, task="transcribe", vad_filter=False)
            segments = list(lazy_segments)
        except (ValueError, RuntimeError, OSError) as error:
            raise SttError("Faster-Whisper transcription failed") from error
        processing_seconds = self.monotonic_seconds() - start_seconds
        return Transcript(
            text="".join(segment.text for segment in segments),
            language=info.language,
            language_probability=info.language_probability,
            audio_duration_seconds=utterance.duration_seconds,
            processing_seconds=processing_seconds,
            real_time_factor=processing_seconds / utterance.duration_seconds,
            model_name=self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            segments=tuple(TranscriptSegment(segment.start, segment.end, segment.text) for segment in segments),
        )
