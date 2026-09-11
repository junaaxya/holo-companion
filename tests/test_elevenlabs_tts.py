import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import numpy as np
import pytest

from holo_companion.llm.base import CancellationHandle
from holo_companion.tts.base import (
    StyleHints,
    SynthesisRequest,
    SynthesisResult,
    SynthesisStatus,
    TTSAudioChunk,
    TtsError,
    TtsErrorKind,
)
from holo_companion.tts.config import load_elevenlabs_tts_config
from holo_companion.tts.elevenlabs import (
    ELEVENLABS_V3_CONVERSATIONAL_MODEL,
    ElevenLabsConfig,
    ElevenLabsTTSProvider,
    EmotionMapper,
    provider_from_config,
)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeClock:
    values: list[float]

    def __call__(self) -> float:
        return self.values.pop(0)


class ChunkByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class MockHttpConnector:
    def __init__(self, handler) -> None:
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.closed = False

    def stream_text_to_dialogue(self, url: str, headers: dict[str, str], payload: dict[str, object]):
        return self.client.stream("POST", url, headers=headers, json=payload)

    async def aclose(self) -> None:
        self.closed = True
        await self.client.aclose()


def pcm16(values: list[int]) -> bytes:
    return np.asarray(values, dtype=np.int16).tobytes()


def http_provider_with_response(
    response: httpx.Response, clock: FakeClock
) -> tuple[ElevenLabsTTSProvider, list[httpx.Request], MockHttpConnector]:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response.request = request
        return response

    config = ElevenLabsConfig(
        api_key="test-secret",
        voice_id="voice-a",
        model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL,
        base_url="https://api.example.test",
        timeout_seconds=5.0,
    )
    connector = MockHttpConnector(handler)
    provider = ElevenLabsTTSProvider(config=config, connector=connector, monotonic_seconds=clock)
    return provider, requests, connector


@pytest.mark.anyio
async def test_elevenlabs_http_provider_streams_chunks_and_preserves_final_pcm() -> None:
    # Given
    body_part1 = pcm16([1, 2, 3])  # 6 bytes (3 samples)
    body_part2 = pcm16([4, 5])     # 4 bytes (2 samples)
    stream = ChunkByteStream([body_part1, body_part2])
    response = httpx.Response(200, stream=stream)
    provider, requests, connector = http_provider_with_response(response, FakeClock([10.0, 10.1, 10.2, 10.3]))

    # When
    events = [event async for event in provider.stream(SynthesisRequest(text="Halo, Master."))]

    # Then
    assert len(requests) == 1
    req = requests[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.example.test/v1/text-to-dialogue/stream?output_format=pcm_24000"
    assert req.headers["xi-api-key"] == "test-secret"
    payload = json.loads(req.content)
    assert payload["model_id"] == ELEVENLABS_V3_CONVERSATIONAL_MODEL
    assert payload["inputs"] == [{"text": "Halo, Master.", "voice_id": "voice-a"}]

    chunks = [event for event in events if isinstance(event, TTSAudioChunk)]
    assert len(chunks) == 2
    assert chunks[0].samples.size == 3
    assert chunks[1].samples.size == 2
    assert chunks[0].sequence == 0
    assert chunks[1].sequence == 1

    result = events[-1]
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.COMPLETED
    assert result.metrics.chunk_count == 2
    assert result.metrics.received_byte_count == 10
    assert result.metrics.generated_audio_seconds == 5 / 24_000
    assert stream.closed


@pytest.mark.anyio
async def test_elevenlabs_http_provider_handles_partial_byte_carry() -> None:
    # Given
    full_pcm = pcm16([10, 20, 30])  # 6 bytes
    # Split unevenly across chunks
    stream = ChunkByteStream([full_pcm[:3], full_pcm[3:]])
    response = httpx.Response(200, stream=stream)
    provider, _requests, _connector = http_provider_with_response(response, FakeClock([1.0, 1.1, 1.2, 1.3]))

    # When
    events = [event async for event in provider.stream(SynthesisRequest(text="Tes"))]

    # Then
    chunks = [event for event in events if isinstance(event, TTSAudioChunk)]
    assert len(chunks) == 2
    # chunk 0 gets first 2 bytes (1 sample), carries 1 byte
    assert chunks[0].samples.size == 1
    # chunk 1 gets remaining 3 bytes + carried 1 byte = 4 bytes (2 samples)
    assert chunks[1].samples.size == 2
    total_samples = sum(c.samples.size for c in chunks)
    assert total_samples == 3


@pytest.mark.anyio
async def test_elevenlabs_http_provider_cancellation_aborts_stream() -> None:
    # Given
    stream = ChunkByteStream([pcm16([1, 2]), pcm16([3, 4])])
    response = httpx.Response(200, stream=stream)
    provider, _requests, _connector = http_provider_with_response(response, FakeClock([1.0, 1.1, 1.2]))
    cancellation = CancellationHandle()

    # When
    events_iter = provider.stream(SynthesisRequest(text="Batal"), cancellation=cancellation)
    chunk1 = await anext(events_iter)
    assert isinstance(chunk1, TTSAudioChunk)
    cancellation.cancel("barge-in")
    result = await anext(events_iter)

    # Then
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.CANCELLED
    assert result.metrics.chunk_count == 1
    assert stream.closed


@pytest.mark.anyio
async def test_elevenlabs_http_provider_status_error_mapped_to_tts_error() -> None:
    # Given
    response = httpx.Response(401, text='{"detail":{"status":"invalid_api_key"}}')
    provider, _requests, _connector = http_provider_with_response(response, FakeClock([1.0]))

    # When / Then
    with pytest.raises(TtsError) as exc_info:
        _ = [event async for event in provider.stream(SynthesisRequest(text="Error"))]

    assert exc_info.value.kind is TtsErrorKind.PROVIDER
    assert exc_info.value.diagnostic is not None
    assert exc_info.value.diagnostic.status_code == 401
    assert exc_info.value.diagnostic.error == "status_error"


@pytest.mark.anyio
async def test_elevenlabs_http_provider_timeout_mapped_to_tts_error() -> None:
    # Given
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout during stream", request=request)

    config = ElevenLabsConfig(
        api_key="test-secret",
        voice_id="voice-a",
        model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL,
        base_url="https://api.example.test",
    )
    connector = MockHttpConnector(handler)
    provider = ElevenLabsTTSProvider(config=config, connector=connector, monotonic_seconds=FakeClock([1.0]))

    # When / Then
    with pytest.raises(TtsError) as exc_info:
        _ = [event async for event in provider.stream(SynthesisRequest(text="Timeout"))]

    assert exc_info.value.kind is TtsErrorKind.PROVIDER
    assert exc_info.value.diagnostic is not None
    assert exc_info.value.diagnostic.stage == "timeout"


def test_provider_from_config_uses_http_streaming_connector() -> None:
    # Given
    config = ElevenLabsConfig(
        api_key="test-secret",
        voice_id="voice-a",
        model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL,
    )

    # When
    provider = provider_from_config(config, lambda: 0.0)

    # Then
    assert hasattr(provider.connector, "stream_text_to_dialogue")
    assert not hasattr(provider.connector, "connect")


def test_emotion_mapper_is_deterministic_preserves_original_dialogue_words() -> None:
    # Given
    mapper = EmotionMapper()
    text = "Ih... ternyata kamu belum tidur juga."

    # When
    neutral = mapper.map_text(text, StyleHints(emotion="neutral", intensity=1.0))
    playful = mapper.map_text(text, StyleHints(emotion="playful", intensity=0.6))

    # Then
    assert neutral.text == text
    assert playful == mapper.map_text(text, StyleHints(emotion="playful", intensity=0.6))
    assert playful.text == "[mischievously] Ih... ternyata kamu belum tidur juga!"
    assert playful.text.removeprefix("[mischievously] ").rstrip("!") == text.rstrip(".")
    assert "shout" not in playful.text.lower()
    assert "loud" not in playful.text.lower()


def test_emotion_mapper_renders_exact_provider_tags_for_requested_emotions() -> None:
    # Given
    mapper = EmotionMapper()
    text = "Halo"
    expected: Final = {
        "happy": "[happily] Halo!",
        "playful": "[mischievously] Halo!",
        "relaxed": "[softly] Halo",
        "concerned": "[worried] Halo",
        "sad": "[sad] Halo...",
        "angry": "[angry] Halo",
        "surprised": "[surprised] Halo!",
        "shy": "[hesitantly] Halo...",
    }

    # When / Then
    assert mapper.map_text(text, StyleHints(emotion="neutral", intensity=0.9)).text == text
    for emotion, rendered_text in expected.items():
        mapped = mapper.map_text(text, StyleHints(emotion=emotion, intensity=0.6))
        assert mapped.text == rendered_text
        assert mapped.text.removeprefix(mapped.text.split("] ", 1)[0] + "] ").rstrip("!.") == text
        assert "[whisper" not in mapped.text
        assert "[sigh" not in mapped.text
        assert "loud" not in mapped.text.lower()
        assert "shout" not in mapped.text.lower()
    with pytest.raises(TtsError, match="unsupported emotion"):
        mapper.map_text(text, StyleHints(emotion="excited", intensity=0.5))


def test_emotion_mapper_uses_single_base_tag_for_low_intensity() -> None:
    # Given
    mapper = EmotionMapper()

    # When / Then
    assert mapper.map_text("Halo", StyleHints(emotion="playful", intensity=0.2)).text == "[mischievously] Halo"
    assert mapper.map_text("Halo", StyleHints(emotion="sad", intensity=0.2)).text == "[sad] Halo"
    assert mapper.map_text("Halo", StyleHints(emotion="playful", intensity=0.5)).text == "[mischievously] Halo!"
    assert mapper.map_text("Halo", StyleHints(emotion="playful", intensity=0.8)).text == "[mischievously] Halo!"
    assert mapper.map_text("Halo", StyleHints(emotion="playful", intensity=0.8)).text.count("[") == 1


def test_emotion_mapper_uses_emotion_aware_high_intensity_without_double_exclamation() -> None:
    # Given
    mapper = EmotionMapper()
    expected: Final = {
        "happy": "[excited] Halo!",
        "playful": "[mischievously] Halo!",
        "relaxed": "[softly] Halo...",
        "concerned": "[worried] Halo...",
        "sad": "[sad] Halo...",
        "angry": "[angry] Halo!",
        "surprised": "[excited] Halo!",
        "shy": "[whispers] Halo...",
    }

    # When / Then
    for emotion, rendered_text in expected.items():
        mapped = mapper.map_text("Halo", StyleHints(emotion=emotion, intensity=0.8)).text
        assert mapped == rendered_text
        assert "!!" not in mapped
        assert "[shout" not in mapped
        assert "[sigh" not in mapped


def test_elevenlabs_config_loads_env_secrets_and_requires_v3_model(tmp_path: Path) -> None:
    # Given
    from pathlib import Path
    config_path = tmp_path / "default.toml"
    config_path.write_text('[tts.elevenlabs]\nbase_url = "https://api.example.test"\ntimeout_seconds = 9.0\n', encoding="utf-8")

    # When
    config = load_elevenlabs_tts_config(config_path, {"ELEVENLABS_API_KEY": "secret", "ELEVENLABS_VOICE_ID": "voice-a", "ELEVENLABS_MODEL_ID": ELEVENLABS_V3_CONVERSATIONAL_MODEL})

    # Then
    assert config.api_key == "secret"
    assert config.voice_id == "voice-a"
    assert config.model_id == ELEVENLABS_V3_CONVERSATIONAL_MODEL
    assert config.base_url == "https://api.example.test"
    assert config.timeout_seconds == 9.0
    with pytest.raises(TtsError, match="eleven_v3_conversational"):
        load_elevenlabs_tts_config(config_path, {"ELEVENLABS_API_KEY": "secret", "ELEVENLABS_VOICE_ID": "voice-a", "ELEVENLABS_MODEL_ID": "wrong"})
