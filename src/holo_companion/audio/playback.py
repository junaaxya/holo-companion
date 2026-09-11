from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Condition, Lock
from typing import Protocol

import anyio
import numpy as np
import sounddevice
from numpy.typing import NDArray

from holo_companion.audio.types import AudioCliError, DeviceIndex
from holo_companion.runtime.playback import PlaybackAdmission

DEFAULT_PLAYBACK_BLOCKSIZE = 512
DEFAULT_MAX_BUFFER_SECONDS = 2.0


class OutputStreamLike(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class PlaybackTimeInfo(Protocol):
    inputBufferAdcTime: float
    outputBufferDacTime: float
    currentTime: float


class PlaybackStatus(Protocol):
    input_underflow: bool
    output_underflow: bool


class OutputStreamFactory(Protocol):
    def __call__(self, *, samplerate: int, channels: int, dtype: str, device: int | None, blocksize: int, callback: Callable[[NDArray[np.float32], int, PlaybackTimeInfo | None, PlaybackStatus | None], None]) -> OutputStreamLike: ...


@dataclass(frozen=True, slots=True)
class PlaybackDiagnostics:
    device_output_underflows: int = 0
    playback_starvations: int = 0
    backpressure_waits: int = 0


@dataclass(slots=True)
class BufferedChunk:
    admission: PlaybackAdmission
    offset: int = 0


@dataclass(slots=True)
class SoundDevicePlaybackSink:
    output_device: DeviceIndex | None = None
    output_factory: OutputStreamFactory = sounddevice.OutputStream
    blocksize: int = DEFAULT_PLAYBACK_BLOCKSIZE
    max_buffer_seconds: float = DEFAULT_MAX_BUFFER_SECONDS
    max_buffer_samples: int | None = None
    on_first_consumed: Callable[[PlaybackAdmission], None] | None = None
    on_samples_consumed: Callable[[NDArray[np.float32], int], None] | None = None
    _condition: Condition = field(default_factory=lambda: Condition(Lock()), init=False)
    _buffer: deque[BufferedChunk] = field(default_factory=deque, init=False)
    _stream: OutputStreamLike | None = field(default=None, init=False)
    _sample_rate_hz: int | None = field(default=None, init=False)
    _queued_samples: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _acknowledged_first_consumption: bool = field(default=False, init=False)
    _diagnostics: PlaybackDiagnostics = field(default_factory=PlaybackDiagnostics, init=False)
    _epoch: int = field(default=0, init=False)
    _waiting_writers: int = field(default=0, init=False)
    _generation_active: bool = field(default=False, init=False)
    _total_consumed_samples: int = field(default=0, init=False)

    @property
    def total_consumed_samples(self) -> int:
        with self._condition:
            return self._total_consumed_samples

    @property
    def diagnostics(self) -> PlaybackDiagnostics:
        return self._diagnostics

    @property
    def waiting_writers(self) -> int:
        with self._condition:
            return self._waiting_writers

    async def write_and_start(self, admission: PlaybackAdmission) -> None:
        await anyio.lowlevel.checkpoint()
        sample_rate_hz = admission.chunk.audio_format.sample_rate_hz
        with self._condition:
            if self._closed:
                raise AudioCliError("playback sink is closed")
            stream = self._stream
            if self._sample_rate_hz != sample_rate_hz:
                self._buffer.clear()
                self._queued_samples = 0
                self._acknowledged_first_consumption = False
                stream = None
                self._sample_rate_hz = sample_rate_hz
            epoch = self._epoch
        try:
            accepted = await anyio.to_thread.run_sync(
                self._wait_for_capacity,
                admission.chunk.samples.size,
                sample_rate_hz,
                epoch,
                abandon_on_cancel=True,
            )
        except anyio.get_cancelled_exc_class():
            self._invalidate_waiters()
            raise
        if not accepted:
            return
        with self._condition:
            if self._closed or self._epoch != epoch:
                return
            self._buffer.append(BufferedChunk(admission))
            self._queued_samples += admission.chunk.samples.size
        if stream is None:
            stream = self.output_factory(
                samplerate=sample_rate_hz,
                channels=1,
                dtype="float32",
                device=None if self.output_device is None else int(self.output_device),
                blocksize=self.blocksize,
                callback=self._callback,
            )
            with self._condition:
                self._stream = stream
        stream.start()

    async def set_generation_active(self, active: bool) -> None:
        with self._condition:
            self._generation_active = active

    async def wait_until_drained(self) -> None:
        while True:
            with self._condition:
                if self._closed or (self._queued_samples == 0 and len(self._buffer) == 0):
                    break
            await anyio.sleep(0.02)
        await anyio.sleep(0.05)

    async def stop(self) -> None:
        with self._condition:
            self._buffer.clear()
            self._queued_samples = 0
            self._acknowledged_first_consumption = False
            self._epoch += 1
            self._generation_active = False
            stream = self._stream
            self._condition.notify_all()
        if stream is not None:
            stream.stop()

    async def aclose(self) -> None:
        with self._condition:
            self._buffer.clear()
            self._queued_samples = 0
            self._acknowledged_first_consumption = False
            self._closed = True
            self._epoch += 1
            self._generation_active = False
            stream = self._stream
            self._stream = None
            self._condition.notify_all()
        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(self, outdata: NDArray[np.float32], frames: int, time_info: PlaybackTimeInfo | None, status: PlaybackStatus | None) -> None:
        del time_info
        if status is not None and status.output_underflow:
            self._count_device_underflow()
        written = 0
        flat = outdata.reshape(frames)
        flat.fill(0.0)
        with self._condition:
            while written < frames and self._buffer:
                item = self._buffer[0]
                samples = item.admission.chunk.samples
                available = samples.size - item.offset
                requested = frames - written
                count = min(available, requested)
                flat[written : written + count] = samples[item.offset : item.offset + count]
                if item.offset == 0 and not self._acknowledged_first_consumption and self.on_first_consumed is not None:
                    self._acknowledged_first_consumption = True
                    self.on_first_consumed(item.admission)
                item.offset += count
                written += count
                self._queued_samples -= count
                self._total_consumed_samples += count
                if item.offset == samples.size:
                    self._buffer.popleft()
            self._condition.notify_all()

            if written == 0 and self._generation_active:
                self._count_starvation()
        if written > 0 and self.on_samples_consumed is not None and self._sample_rate_hz is not None:
            self.on_samples_consumed(flat[:written].copy(), self._sample_rate_hz)

    def _count_device_underflow(self) -> None:
        self._diagnostics = PlaybackDiagnostics(
            self._diagnostics.device_output_underflows + 1,
            self._diagnostics.playback_starvations,
            self._diagnostics.backpressure_waits,
        )

    def _count_starvation(self) -> None:
        self._diagnostics = PlaybackDiagnostics(
            self._diagnostics.device_output_underflows,
            self._diagnostics.playback_starvations + 1,
            self._diagnostics.backpressure_waits,
        )

    def _wait_for_capacity(self, samples: int, sample_rate_hz: int, epoch: int) -> bool:
        with self._condition:
            capacity = self._capacity_samples(sample_rate_hz)
            while not self._closed and self._epoch == epoch and self._queued_samples + samples > capacity:
                self._diagnostics = PlaybackDiagnostics(
                    self._diagnostics.device_output_underflows,
                    self._diagnostics.playback_starvations,
                    self._diagnostics.backpressure_waits + 1,
                )
                self._waiting_writers += 1
                try:
                    self._condition.wait()
                finally:
                    self._waiting_writers -= 1
            return not self._closed and self._epoch == epoch

    def _invalidate_waiters(self) -> None:
        with self._condition:
            self._epoch += 1
            self._condition.notify_all()

    def _capacity_samples(self, sample_rate_hz: int) -> int:
        if self.max_buffer_samples is not None:
            return self.max_buffer_samples
        return int(sample_rate_hz * self.max_buffer_seconds)
