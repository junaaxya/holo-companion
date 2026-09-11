from dataclasses import dataclass
from typing import Protocol

from holo_companion.tts.base import TTSAudioChunk


@dataclass(frozen=True, slots=True)
class PlaybackAdmission:
    turn_id: int
    generation_id: int
    chunk: TTSAudioChunk


class PlaybackSink(Protocol):
    async def write_and_start(self, admission: PlaybackAdmission) -> None: ...

    async def stop(self) -> None: ...

    async def aclose(self) -> None: ...

    async def set_generation_active(self, active: bool) -> None: ...

    async def wait_until_drained(self) -> None: ...
