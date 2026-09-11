from dataclasses import dataclass, field

import numpy as np
import pytest
import anyio

from holo_companion.audio.playback import SoundDevicePlaybackSink
from holo_companion.audio.types import AudioCliError
from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.tts.base import TTSAudioChunk, TtsAudioFormat


def chunk(sequence: int, values: tuple[float, ...]) -> TTSAudioChunk:
    return TTSAudioChunk.from_samples(np.asarray(values, dtype=np.float32), TtsAudioFormat(sample_rate_hz=24_000), sequence, 0.0)


@dataclass(slots=True)
class FakeOutputStream:
    callback: object
    started: bool = False
    stopped: bool = False
    closed: bool = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def consume(self, frames: int) -> np.ndarray:
        outdata = np.zeros((frames, 1), dtype=np.float32)
        assert callable(self.callback)
        self.callback(outdata, frames, None, None)
        return outdata.reshape(frames)


@dataclass(slots=True)
class FakeOutputFactory:
    streams: list[FakeOutputStream] = field(default_factory=list)

    def __call__(self, *, samplerate: int, channels: int, dtype: str, device: int | None, blocksize: int, callback):
        del samplerate, channels, dtype, device, blocksize
        stream = FakeOutputStream(callback)
        self.streams.append(stream)
        return stream


def admission(sequence: int, values: tuple[float, ...]) -> PlaybackAdmission:
    return PlaybackAdmission(1, 1, chunk(sequence, values))


@pytest.mark.anyio
async def test_playback_preserves_chunk_order_and_acknowledges_callback_consumption() -> None:
    # Given
    factory = FakeOutputFactory()
    consumed: list[int] = []
    sink = SoundDevicePlaybackSink(output_factory=factory, on_first_consumed=lambda item: consumed.append(item.chunk.sequence))

    # When
    await sink.write_and_start(admission(0, (0.1, 0.2)))
    await sink.write_and_start(admission(1, (0.3, 0.4)))
    first = factory.streams[0].consume(3)
    second = factory.streams[0].consume(1)

    # Then
    assert first.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert second.tolist() == pytest.approx([0.4])
    assert consumed == [0]


@pytest.mark.anyio
async def test_playback_reports_consumed_pcm_to_echo_reference_hook() -> None:
    # Given
    factory = FakeOutputFactory()
    consumed: list[tuple[np.ndarray, int]] = []
    sink = SoundDevicePlaybackSink(
        output_factory=factory,
        on_samples_consumed=lambda samples, sample_rate_hz: consumed.append((samples, sample_rate_hz)),
    )
    await sink.write_and_start(admission(0, (0.1, 0.2)))

    # When
    factory.streams[0].consume(2)

    # Then
    assert len(consumed) == 1
    np.testing.assert_allclose(consumed[0][0], np.asarray([0.1, 0.2], dtype=np.float32))
    assert consumed[0][1] == 24_000


@pytest.mark.anyio
async def test_playback_stop_clears_stale_audio_and_reports_underrun() -> None:
    # Given
    factory = FakeOutputFactory()
    sink = SoundDevicePlaybackSink(output_factory=factory)
    await sink.write_and_start(admission(0, (0.1, 0.2, 0.3)))

    # When
    await sink.stop()
    output = factory.streams[0].consume(3)

    # Then
    assert output.tolist() == [0.0, 0.0, 0.0]
    assert sink.diagnostics.playback_starvations == 0
    assert factory.streams[0].stopped


@pytest.mark.anyio
async def test_playback_backend_output_underflow_is_counted_without_crashing() -> None:
    # Given
    factory = FakeOutputFactory()
    sink = SoundDevicePlaybackSink(output_factory=factory)
    await sink.write_and_start(admission(0, (0.1,)))

    # When
    status = type("Status", (), {"input_underflow": False, "output_underflow": True})()
    outdata = np.zeros((1, 1), dtype=np.float32)
    factory.streams[0].callback(outdata, 1, None, status)

    # Then
    assert sink.diagnostics.device_output_underflows == 1


@pytest.mark.anyio
async def test_active_generation_silence_is_playback_starvation() -> None:
    # Given
    factory = FakeOutputFactory()
    sink = SoundDevicePlaybackSink(output_factory=factory)
    await sink.write_and_start(admission(0, (0.1,)))
    await sink.set_generation_active(True)
    factory.streams[0].consume(1)

    # When
    factory.streams[0].consume(1)

    # Then
    assert sink.diagnostics.playback_starvations == 1
    assert sink.diagnostics.device_output_underflows == 0


@pytest.mark.anyio
async def test_playback_close_releases_stream_and_rejects_later_write() -> None:
    # Given
    factory = FakeOutputFactory()
    sink = SoundDevicePlaybackSink(output_factory=factory)
    await sink.write_and_start(admission(0, (0.1,)))

    # When
    await sink.aclose()

    # Then
    assert factory.streams[0].closed
    with pytest.raises(AudioCliError, match="closed"):
        await sink.write_and_start(admission(1, (0.2,)))


@pytest.mark.anyio
async def test_playback_buffer_is_bounded() -> None:
    # Given
    sink = SoundDevicePlaybackSink(output_factory=FakeOutputFactory(), max_buffer_samples=2)
    await sink.write_and_start(admission(0, (0.1, 0.2)))

    # When
    writer_started = anyio.Event()
    writer_finished = anyio.Event()

    async def write_second() -> None:
        writer_started.set()
        await sink.write_and_start(admission(1, (0.3,)))
        writer_finished.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(write_second)
        await writer_started.wait()
        while sink.waiting_writers == 0:
            await anyio.lowlevel.checkpoint()
        assert not writer_finished.is_set()
        sink.output_factory.streams[0].consume(2)

    # Then
    assert writer_finished.is_set()
    assert sink.diagnostics.backpressure_waits > 0


@pytest.mark.anyio
async def test_playback_stop_wakes_blocked_writer_and_prevents_stale_resume() -> None:
    # Given
    factory = FakeOutputFactory()
    sink = SoundDevicePlaybackSink(output_factory=factory, max_buffer_samples=2)
    await sink.write_and_start(admission(0, (0.1, 0.2)))
    completed = anyio.Event()

    async def write_stale() -> None:
        await sink.write_and_start(admission(1, (0.3,)))
        completed.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(write_stale)
        while sink.waiting_writers == 0:
            await anyio.lowlevel.checkpoint()
        await sink.stop()

    # Then
    assert completed.is_set()
    assert factory.streams[0].consume(2).tolist() == [0.0, 0.0]


@pytest.mark.anyio
async def test_playback_close_wakes_blocked_writer() -> None:
    # Given
    sink = SoundDevicePlaybackSink(output_factory=FakeOutputFactory(), max_buffer_samples=2)
    await sink.write_and_start(admission(0, (0.1, 0.2)))

    async def write_second() -> None:
        await sink.write_and_start(admission(1, (0.3,)))

    async with anyio.create_task_group() as tg:
        tg.start_soon(write_second)
        while sink.waiting_writers == 0:
            await anyio.lowlevel.checkpoint()
        await sink.aclose()
