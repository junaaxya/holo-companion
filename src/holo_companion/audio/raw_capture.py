from pathlib import Path

import soundfile

from holo_companion.audio.types import AudioCliError, AudioFrame, CAPTURE_SAMPLE_RATE


class RawCaptureDumper:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = soundfile.SoundFile(
                path,
                mode="w",
                samplerate=int(CAPTURE_SAMPLE_RATE),
                channels=1,
                format="WAV",
                subtype="PCM_16",
            )
        except soundfile.SoundFileError as error:
            raise AudioCliError(f"failed to open raw capture dump: {path}") from error
        self.path = path

    def write_frame(self, frame: AudioFrame) -> None:
        try:
            self._file.write(frame.samples)
        except soundfile.SoundFileError as error:
            raise AudioCliError(f"failed to write raw capture dump: {self.path}") from error

    def close(self) -> None:
        self._file.close()
