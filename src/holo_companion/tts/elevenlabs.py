import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import urlsplit, urlunsplit

import anyio
import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException

from holo_companion.llm.base import CancellationHandle
from holo_companion.tts.base import SynthesisMetrics, SynthesisRequest, SynthesisResult, SynthesisStatus, TTSAudioChunk, TtsAudioFormat, TtsError, TtsErrorKind, TtsStreamEvent, real_time_factor
from holo_companion.tts._elevenlabs_diagnostics import config_error, handshake_status_code, provider_diagnostic_error, sanitize_message, socket_close_code, with_model
from holo_companion.tts._elevenlabs_emotion import EmotionMapper
from holo_companion.tts._elevenlabs_ws import ElevenLabsWebSocket, ElevenLabsWebSocketConnector, WebsocketsConnector, close_socket_frame, decode_audio, input_frame, parse_ws_event, provider_error, text_to_dialogue_url, voice_registration

ELEVENLABS_V3_CONVERSATIONAL_MODEL: Final = "eleven_v3_conversational"
ELEVENLABS_PROVIDER: Final = "elevenlabs"
ELEVENLABS_SAMPLE_RATE_HZ: Final = 24_000


class Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str
    base_url: str = "https://api.elevenlabs.io"
    timeout_seconds: float = 30.0
    stability: float = 0.40

    def __post_init__(self) -> None:
        if self.api_key.strip() == "":
            raise config_error("ELEVENLABS_API_KEY is not configured", ELEVENLABS_V3_CONVERSATIONAL_MODEL)
        object.__setattr__(self, "api_key", self.api_key.strip())
        if self.voice_id.strip() == "":
            raise config_error("ELEVENLABS_VOICE_ID is not configured", ELEVENLABS_V3_CONVERSATIONAL_MODEL)
        object.__setattr__(self, "voice_id", self.voice_id.strip())
        if self.model_id.strip() != ELEVENLABS_V3_CONVERSATIONAL_MODEL:
            raise config_error("ELEVENLABS_MODEL_ID must be exactly eleven_v3_conversational", ELEVENLABS_V3_CONVERSATIONAL_MODEL)
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        if self.timeout_seconds <= 0.0:
            raise config_error("ElevenLabs timeout_seconds must be greater than 0", ELEVENLABS_V3_CONVERSATIONAL_MODEL)
        if not np.isfinite(self.stability) or not 0.0 <= self.stability <= 1.0:
            raise config_error("ElevenLabs stability must be between 0.0 and 1.0", ELEVENLABS_V3_CONVERSATIONAL_MODEL)


@dataclass(frozen=True, slots=True)
class ElevenLabsTTSProvider:
    config: ElevenLabsConfig
    connector: ElevenLabsWebSocketConnector
    monotonic_seconds: Clock
    emotion_mapper: EmotionMapper = EmotionMapper()
    _active_socket: ElevenLabsWebSocket | None = None

    async def stream(self, request: SynthesisRequest, cancellation: CancellationHandle | None = None) -> AsyncIterator[TtsStreamEvent]:
        handle = CancellationHandle() if cancellation is None else cancellation
        if handle.is_cancelled():
            raise TtsError(TtsErrorKind.CANCELLED, "cancelled before ElevenLabs synthesis")
        mapped = self.emotion_mapper.map_text(request.text, request.style)
        sequence = 0
        byte_count = 0
        sample_count = 0
        first_audio_seconds: float | None = None
        carry = b""
        final_audio_for_turn = False
        final_message = False
        audio_format = TtsAudioFormat(sample_rate_hz=ELEVENLABS_SAMPLE_RATE_HZ)
        start = self.monotonic_seconds()
        socket: ElevenLabsWebSocket | None = None
        try:
            try:
                with anyio.fail_after(self.config.timeout_seconds):
                    socket = await self.connector.connect(text_to_dialogue_url(self.config.base_url, self.config.model_id), {"xi-api-key": self.config.api_key})
            except TimeoutError as error:
                raise provider_diagnostic_error("timeout", "websocket_handshake", "timeout", self.config.model_id) from error
            except (InvalidHandshake, WebSocketException) as error:
                raise provider_diagnostic_error("websocket_handshake", "transport_error", "handshake", self.config.model_id, status_code=handshake_status_code(error)) from error
            object.__setattr__(self, "_active_socket", socket)
            try:
                with anyio.fail_after(self.config.timeout_seconds):
                    await socket.send(json.dumps(voice_registration(self.config.voice_id, {"stability": self.config.stability}), separators=(",", ":")))
            except TimeoutError as error:
                raise provider_diagnostic_error("timeout", "registration", "timeout", self.config.model_id) from error
            except WebSocketException as error:
                raise provider_diagnostic_error("registration", "transport_error", "websocket", self.config.model_id) from error
            try:
                with anyio.fail_after(self.config.timeout_seconds):
                    await socket.send(json.dumps(input_frame(mapped.text, self.config.voice_id), separators=(",", ":")))
            except TimeoutError as error:
                raise provider_diagnostic_error("timeout", "input_send", "timeout", self.config.model_id) from error
            except WebSocketException as error:
                raise provider_diagnostic_error("input_send", "transport_error", "websocket", self.config.model_id) from error
            try:
                with anyio.fail_after(self.config.timeout_seconds):
                    await socket.send(json.dumps(close_socket_frame(), separators=(",", ":")))
            except TimeoutError as error:
                raise provider_diagnostic_error("timeout", "close_socket", "timeout", self.config.model_id) from error
            except WebSocketException as error:
                raise provider_diagnostic_error("close_socket", "transport_error", "websocket", self.config.model_id) from error
            while not final_message:
                if handle.is_cancelled():
                    break
                try:
                    with anyio.fail_after(self.config.timeout_seconds):
                        raw_message = await socket.recv()
                except TimeoutError as error:
                    raise provider_diagnostic_error("timeout", "receive", "timeout", self.config.model_id) from error
                except (ConnectionClosed, StopAsyncIteration) as error:
                    raise provider_diagnostic_error("socket_close", "connection_closed", "websocket", self.config.model_id, close_code=socket_close_code(error)) from error
                try:
                    event = parse_ws_event(raw_message)
                except TtsError as error:
                    raise with_model(error, self.config.model_id) from error
                if event.error is not None:
                    message = None if event.error_message is None else sanitize_message(event.error_message, self.config.api_key, self.config.voice_id, mapped.text)
                    raise provider_error("provider_frame", event.error, "protocol", model=self.config.model_id, status_code=event.error_code, message=message)
                if event.audio is not None:
                    try:
                        decoded = decode_audio(event.audio)
                    except TtsError as error:
                        raise with_model(error, self.config.model_id) from error
                    byte_count += len(decoded)
                    data = carry + decoded
                    carry = data[-1:] if len(data) % 2 == 1 else b""
                    complete = data[:-1] if carry else data
                    if complete != b"":
                        try:
                            samples = pcm16_to_float32(complete)
                        except TtsError as error:
                            raise provider_diagnostic_error("audio_decode", "malformed_pcm", "audio", self.config.model_id) from error
                        now = self.monotonic_seconds()
                        if first_audio_seconds is None:
                            first_audio_seconds = now - start
                        chunk = TTSAudioChunk.from_samples(samples=samples, audio_format=audio_format, sequence=sequence, start_seconds=sample_count / ELEVENLABS_SAMPLE_RATE_HZ)
                        sequence += 1
                        sample_count += samples.size
                        yield chunk
                final_audio_for_turn = final_audio_for_turn or event.is_final_audio_for_turn
                final_message = final_message or event.is_final
        finally:
            if socket is not None:
                await socket.aclose()
                object.__setattr__(self, "_active_socket", None)
        if carry:
            raise provider_diagnostic_error("audio_decode", "malformed_pcm", "audio", self.config.model_id)
        if sequence == 0:
            raise provider_diagnostic_error("audio_decode", "empty_audio", "audio", self.config.model_id)
        if not handle.is_cancelled() and (not final_audio_for_turn or not final_message):
            raise provider_diagnostic_error("receive", "missing_final_message", "protocol", self.config.model_id)
        synthesis_seconds = self.monotonic_seconds() - start
        generated_seconds = sample_count / ELEVENLABS_SAMPLE_RATE_HZ
        yield SynthesisResult(
            request=request,
            status=SynthesisStatus.CANCELLED if handle.is_cancelled() else SynthesisStatus.COMPLETED,
            metrics=SynthesisMetrics(
                request_started_at_seconds=start,
                time_to_first_audio_seconds=first_audio_seconds,
                synthesis_seconds=synthesis_seconds,
                generated_audio_seconds=generated_seconds,
                real_time_factor=real_time_factor(synthesis_seconds, generated_seconds),
                chunk_count=sequence,
                received_byte_count=byte_count,
                provider=ELEVENLABS_PROVIDER,
                model=self.config.model_id,
            ),
        )

    async def aclose(self) -> None:
        if self._active_socket is not None:
            await self._active_socket.aclose()
            object.__setattr__(self, "_active_socket", None)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2 == 1:
        raise TtsError(TtsErrorKind.PROVIDER, "PCM16 chunk must contain whole samples")
    int_samples = np.frombuffer(data, dtype="<i2")
    return np.ascontiguousarray(int_samples.astype(np.float32) / 32_768.0)


def provider_from_config(config: ElevenLabsConfig, monotonic_seconds: Clock) -> ElevenLabsTTSProvider:
    return ElevenLabsTTSProvider(config=config, connector=WebsocketsConnector(), monotonic_seconds=monotonic_seconds)


def normalize_base_url(raw_url: str) -> str:
    normalized = raw_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise TtsError(TtsErrorKind.CONFIG, "ElevenLabs base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise TtsError(TtsErrorKind.CONFIG, "ElevenLabs base_url must not contain credentials")
    if parsed.query != "" or parsed.fragment != "":
        raise TtsError(TtsErrorKind.CONFIG, "ElevenLabs base_url must not contain query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
