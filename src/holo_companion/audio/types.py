from dataclasses import dataclass
from pathlib import Path
from typing import Final, NewType

DeviceIndex = NewType("DeviceIndex", int)
Frames = NewType("Frames", int)
SampleRateHz = NewType("SampleRateHz", int)
Seconds = NewType("Seconds", float)

CAPTURE_CHANNELS: Final = 1
CAPTURE_DTYPE: Final = "float32"
CAPTURE_SAMPLE_RATE: Final = SampleRateHz(16_000)
LOW_VOLUME_PEAK_THRESHOLD: Final = 0.02
WAV_FORMAT: Final = "WAV"
WAV_SUBTYPE: Final = "PCM_16"


@dataclass(frozen=True, slots=True)
class AudioCliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: DeviceIndex
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    requested: str
    resolved: DeviceInfo


@dataclass(frozen=True, slots=True)
class DeviceDefaults:
    input_selection: DeviceSelection
    output_selection: DeviceSelection


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    seconds: Seconds
    output_path: Path
    requested_input_device: str
    overwrite: bool


@dataclass(frozen=True, slots=True)
class CaptureFormat:
    samplerate_hz: SampleRateHz = CAPTURE_SAMPLE_RATE
    channels: int = CAPTURE_CHANNELS
    dtype: str = CAPTURE_DTYPE
    wav_format: str = WAV_FORMAT
    wav_subtype: str = WAV_SUBTYPE


@dataclass(frozen=True, slots=True)
class AudioDiagnostics:
    frames: Frames
    duration_seconds: float
    peak_abs_amplitude: float
    rms: float
    clipping_samples: int
    clipping: bool
    low_volume_threshold: float
    low_volume_status: str


@dataclass(frozen=True, slots=True)
class CaptureResult:
    requested_input_device: str
    resolved_input_device: DeviceInfo
    output_path: Path
    format: CaptureFormat
    diagnostics: AudioDiagnostics
