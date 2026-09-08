import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias
from urllib.parse import parse_qs, urlsplit

import anyio
import numpy as np
import pytest

from holo_companion.llm.base import CancellationHandle
from holo_companion.tts.base import StyleHints, SynthesisRequest, SynthesisResult, SynthesisStatus, TTSAudioChunk, TtsError, TtsErrorKind
from holo_companion.tts.config import load_elevenlabs_tts_config
from holo_companion.tts.elevenlabs import ELEVENLABS_V3_CONVERSATIONAL_MODEL, ElevenLabsConfig, ElevenLabsTTSProvider, EmotionMapper


Frame: TypeAlias = str | bytes | Literal["timeout"]


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeClock:
    values: list[float]

    def __call__(self) -> float:
        return self.values.pop(0)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeSocket:
    frames: list[Frame]
    sent: list[str] = field(default_factory=list)
    closed: bool = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if len(self.frames) == 0:
            raise StopAsyncIteration
        frame = self.frames.pop(0)
        match frame:
            case "timeout":
                await anyio.sleep(1.0)
                return "{}"
            case str() | bytes():
                return frame

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeConnector:
    socket: FakeSocket
    timeout: bool = False
    urls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str]] = field(default_factory=list)

    async def connect(self, url: str, headers: Mapping[str, str]) -> FakeSocket:
        self.urls.append(url)
        self.headers.append(dict(headers))
        if self.timeout:
            await anyio.sleep(1.0)
        return self.socket


def pcm16(values: list[int]) -> bytes:
    return np.asarray(values, dtype=np.int16).tobytes()


def audio_frame(data: bytes) -> str:
    return json.dumps({"audio": base64.b64encode(data).decode("ascii")})


def final_audio_frame() -> str:
    return json.dumps({"is_final_audio_for_turn": True})


def final_frame() -> str:
    return json.dumps({"is_final": True})


def provider_with_socket(socket: FakeSocket, clock: FakeClock, timeout_seconds: float = 7.0) -> tuple[ElevenLabsTTSProvider, FakeConnector]:
    config = ElevenLabsConfig(api_key="test-secret", voice_id="voice-a", model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL, base_url="https://api.example.test", timeout_seconds=timeout_seconds)
    connector = FakeConnector(socket=socket)
    return ElevenLabsTTSProvider(config=config, connector=connector, monotonic_seconds=clock), connector


@pytest.mark.anyio
async def test_elevenlabs_provider_uses_text_to_dialogue_websocket_url_query_and_header() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([0])), final_audio_frame(), final_frame()])
    provider, connector = provider_with_socket(socket, FakeClock([10.0, 10.2, 10.4]))

    # When
    _events = [event async for event in provider.stream(SynthesisRequest(text="Halo"))]

    # Then
    parsed = urlsplit(connector.urls[0])
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.example.test"
    assert parsed.path == "/v1/text-to-dialogue/stream-input"
    assert parse_qs(parsed.query) == {"model_id": [ELEVENLABS_V3_CONVERSATIONAL_MODEL], "output_format": ["pcm_24000"], "language_code": ["id"]}
    assert connector.headers[0] == {"xi-api-key": "test-secret"}


@pytest.mark.anyio
async def test_elevenlabs_provider_sends_voice_registration_then_inputs_with_flush() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([1])), final_audio_frame(), final_frame()])
    provider, _connector = provider_with_socket(socket, FakeClock([1.0, 1.1, 1.2]))
    request = SynthesisRequest(text="Ih... ternyata kamu belum tidur juga.", style=StyleHints(emotion="playful", intensity=0.6))

    # When
    _events = [event async for event in provider.stream(request)]

    # Then
    first = json.loads(socket.sent[0])
    second = json.loads(socket.sent[1])
    third = json.loads(socket.sent[2])
    assert first == {"voices": ["voice-a"], "voice_settings": {"stability": 0.4}}
    assert second == {"inputs": [{"text": "[mischievously] Ih... ternyata kamu belum tidur juga!", "voice_id": "voice-a"}], "flush": True}
    assert third == {"close_socket": True}
    assert all("voice-a" not in message for message in socket.sent[2:])


@pytest.mark.anyio
async def test_elevenlabs_provider_omits_voice_settings_when_mapper_returns_none() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([1])), final_audio_frame(), final_frame()])
    provider, _connector = provider_with_socket(socket, FakeClock([1.0, 1.1, 1.2]))

    # When
    _events = [event async for event in provider.stream(SynthesisRequest(text="Halo", style=StyleHints(emotion="neutral")))]

    # Then
    assert json.loads(socket.sent[0]) == {"voices": ["voice-a"], "voice_settings": {"stability": 0.4}}


@pytest.mark.anyio
async def test_elevenlabs_provider_decodes_base64_pcm_with_carry_and_terminal_metrics() -> None:
    # Given
    body = pcm16([0, 0, 16_384])
    socket = FakeSocket([audio_frame(body[:1]), audio_frame(body[1:2]), audio_frame(body[2:]), final_audio_frame(), final_frame()])
    provider, _connector = provider_with_socket(socket, FakeClock([10.0, 10.2, 10.3, 10.4, 10.6]))

    # When
    events = [event async for event in provider.stream(SynthesisRequest(text="Halo"))]

    # Then
    assert isinstance(events[0], TTSAudioChunk)
    assert isinstance(events[1], TTSAudioChunk)
    np.testing.assert_array_equal(events[0].samples, np.asarray([0.0], dtype=np.float32))
    np.testing.assert_allclose(events[1].samples, np.asarray([0.0, 0.5], dtype=np.float32), rtol=0.0, atol=1e-6)
    result = events[-1]
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.COMPLETED
    assert result.metrics.request_started_at_seconds == 10.0
    assert result.metrics.time_to_first_audio_seconds == 0.1999999999999993
    assert result.metrics.synthesis_seconds == 0.40000000000000036
    assert result.metrics.generated_audio_seconds == 3 / 24_000
    assert result.metrics.real_time_factor == pytest.approx(3_200.0)
    assert result.metrics.chunk_count == 2
    assert result.metrics.received_byte_count == 6
    assert result.metrics.provider == "elevenlabs"
    assert result.metrics.model == ELEVENLABS_V3_CONVERSATIONAL_MODEL
    assert socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_rejects_protocol_error_and_close_before_final() -> None:
    # Given / When / Then
    error_socket = FakeSocket([json.dumps({"error": "quota_exceeded", "message": "No credits for test-secret", "code": 429})])
    error_provider, _connector = provider_with_socket(error_socket, FakeClock([1.0]))
    with pytest.raises(TtsError) as protocol_error:
        _events = [event async for event in error_provider.stream(SynthesisRequest(text="Halo"))]
    assert protocol_error.value.kind is TtsErrorKind.PROVIDER
    assert protocol_error.value.diagnostic is not None
    assert protocol_error.value.diagnostic.stage == "provider_frame"
    assert protocol_error.value.diagnostic.error == "provider_error_frame"
    assert protocol_error.value.diagnostic.status_code == 429
    assert protocol_error.value.diagnostic.message == "No credits for [redacted]"
    assert error_socket.closed

    closed_socket = FakeSocket([audio_frame(pcm16([1]))])
    closed_provider, _connector = provider_with_socket(closed_socket, FakeClock([1.0, 1.1]))
    with pytest.raises(TtsError) as closed_error:
        _events = [event async for event in closed_provider.stream(SynthesisRequest(text="Halo"))]
    assert closed_error.value.diagnostic is not None
    assert closed_error.value.diagnostic.stage == "socket_close"
    assert closed_error.value.diagnostic.error == "connection_closed"
    assert closed_socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_maps_connection_and_receive_timeout_safely() -> None:
    # Given
    config = ElevenLabsConfig(api_key="test-secret", voice_id="voice-a", model_id=ELEVENLABS_V3_CONVERSATIONAL_MODEL, base_url="https://api.example.test", timeout_seconds=0.01)
    connect_socket = FakeSocket([])
    connect_provider = ElevenLabsTTSProvider(config=config, connector=FakeConnector(socket=connect_socket, timeout=True), monotonic_seconds=FakeClock([1.0]))
    receive_socket = FakeSocket(["timeout"])
    receive_provider, _connector = provider_with_socket(receive_socket, FakeClock([1.0]), timeout_seconds=0.01)

    # When / Then
    with pytest.raises(TtsError) as connect_timeout:
        _events = [event async for event in connect_provider.stream(SynthesisRequest(text="Halo"))]
    assert connect_timeout.value.kind is TtsErrorKind.PROVIDER
    assert "test-secret" not in str(connect_timeout.value)
    assert not connect_socket.closed

    with pytest.raises(TtsError) as timeout_error:
        _events = [event async for event in receive_provider.stream(SynthesisRequest(text="Halo"))]
    assert timeout_error.value.kind is TtsErrorKind.PROVIDER
    assert timeout_error.value.diagnostic is not None
    assert timeout_error.value.diagnostic.stage == "timeout"
    assert timeout_error.value.diagnostic.error == "receive"
    assert receive_socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_maps_invalid_json_and_base64_to_safe_provider_error() -> None:
    # Given / When / Then
    for frame in ("{", json.dumps({"audio": "not base64"})):
        socket = FakeSocket([frame])
        provider, _connector = provider_with_socket(socket, FakeClock([1.0]))
        with pytest.raises(TtsError) as raised:
            _events = [event async for event in provider.stream(SynthesisRequest(text="Halo"))]
        assert raised.value.kind is TtsErrorKind.PROVIDER
        assert "test-secret" not in str(raised.value)
        assert socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_cancels_before_connection() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([1])), final_audio_frame(), final_frame()])
    provider, connector = provider_with_socket(socket, FakeClock([1.0]))
    cancellation = CancellationHandle()
    cancellation.cancel("barge-in")

    # When / Then
    with pytest.raises(TtsError) as raised:
        _events = [event async for event in provider.stream(SynthesisRequest(text="Halo"), cancellation)]
    assert raised.value.kind is TtsErrorKind.CANCELLED
    assert connector.urls == []
    assert not socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_mid_stream_cancellation_closes_socket_and_returns_cancelled_result() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([1, 2])), audio_frame(pcm16([3, 4])), final_audio_frame(), final_frame()])
    provider, _connector = provider_with_socket(socket, FakeClock([1.0, 1.1, 1.3]))
    cancellation = CancellationHandle()

    # When
    events = provider.stream(SynthesisRequest(text="Halo"), cancellation)
    first = await anext(events)
    cancellation.cancel("barge-in")
    result = await anext(events)

    # Then
    assert isinstance(first, TTSAudioChunk)
    assert isinstance(result, SynthesisResult)
    assert result.status is SynthesisStatus.CANCELLED
    assert result.metrics.chunk_count == 1
    assert result.metrics.received_byte_count == 4
    assert socket.closed


@pytest.mark.anyio
async def test_elevenlabs_provider_aclose_closes_active_socket() -> None:
    # Given
    socket = FakeSocket([audio_frame(pcm16([1])), final_audio_frame(), final_frame()])
    provider, _connector = provider_with_socket(socket, FakeClock([1.0, 1.1]))

    # When
    events = provider.stream(SynthesisRequest(text="Halo"))
    _first = await anext(events)
    await provider.aclose()

    # Then
    assert socket.closed


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
