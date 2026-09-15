import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx
import numpy as np

from holo_companion.llm.base import CancellationHandle
from holo_companion.tts.base import (
    SynthesisMetrics,
    SynthesisRequest,
    SynthesisResult,
    SynthesisStatus,
    TTSAudioChunk,
    TtsAudioFormat,
    TtsError,
    TtsErrorKind,
    TtsStreamEvent,
    real_time_factor,
)
from holo_companion.tts._elevenlabs_diagnostics import (
    config_error,
    is_quota_exceeded,
    provider_diagnostic_error,
    sanitize_message,
    with_model,
)
from holo_companion.tts._elevenlabs_emotion import EmotionMapper
from holo_companion.tts._elevenlabs_http import DefaultHttpConnector, ElevenLabsHttpConnector
from holo_companion.tts._elevenlabs_ws import (
    ElevenLabsWebSocket,
    ElevenLabsWebSocketConnector,
    WebsocketsConnector,
    decode_audio,
    provider_error,
    text_to_dialogue_url,
)

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
    connector: ElevenLabsHttpConnector
    monotonic_seconds: Clock
    emotion_mapper: EmotionMapper = EmotionMapper()

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
        audio_format = TtsAudioFormat(sample_rate_hz=ELEVENLABS_SAMPLE_RATE_HZ)
        start = self.monotonic_seconds()
        http_status: int | None = None
        content_type: str | None = None
        stream_completed_normally = False

        url = f"{self.config.base_url.rstrip('/')}/v1/text-to-dialogue/stream?output_format=pcm_24000"
        headers = {"xi-api-key": self.config.api_key}
        payload = {
            "model_id": self.config.model_id,
            "inputs": [{"text": mapped.text, "voice_id": self.config.voice_id}],
        }

        try:
            stream_ctx = self.connector.stream_text_to_dialogue(url, headers, payload)
            async with stream_ctx as response:
                http_status = response.status_code
                content_type = response.headers.get("content-type")
                if response.status_code >= 400:
                    body = await response.aread()
                    error_msg = body.decode("utf-8", errors="replace")[:160]
                    if is_quota_exceeded(body):
                        raise provider_error(
                            "http_request",
                            "quota_exceeded",
                            "http",
                            model=self.config.model_id,
                            status_code=response.status_code,
                            message="ElevenLabs quota exhausted",
                        )
                    raise provider_error(
                        "http_request",
                        "status_error",
                        "http",
                        model=self.config.model_id,
                        status_code=response.status_code,
                        message=sanitize_message(error_msg, self.config.api_key, self.config.voice_id, mapped.text),
                    )
                async for raw_bytes in response.aiter_bytes():
                    if handle.is_cancelled():
                        break
                    if not raw_bytes:
                        continue
                    byte_count += len(raw_bytes)
                    data = carry + raw_bytes
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
                        chunk = TTSAudioChunk.from_samples(
                            samples=samples,
                            audio_format=audio_format,
                            sequence=sequence,
                            start_seconds=sample_count / ELEVENLABS_SAMPLE_RATE_HZ,
                        )
                        sequence += 1
                        sample_count += samples.size
                        yield chunk
                stream_completed_normally = True
        except httpx.TimeoutException as error:
            raise provider_diagnostic_error("timeout", "http_stream", "timeout", self.config.model_id) from error
        except httpx.HTTPError as error:
            raise provider_diagnostic_error("http_stream", "transport_error", "http", self.config.model_id) from error

        if carry:
            raise provider_diagnostic_error("audio_decode", "malformed_pcm", "audio", self.config.model_id)
        if sequence == 0 and not handle.is_cancelled():
            from holo_companion.tts.base import TtsProviderDiagnostic
            raise TtsError(
                TtsErrorKind.PROVIDER,
                "ElevenLabs TTS provider failure",
                TtsProviderDiagnostic(
                    stage="audio_decode",
                    status_code=http_status,
                    error="empty_audio",
                    error_class="audio",
                    model=self.config.model_id,
                    content_type=content_type,
                    received_byte_count=byte_count,
                    decoded_sample_count=sample_count,
                    stream_completed_normally=stream_completed_normally,
                ),
            )

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
        if hasattr(self.connector, "aclose"):
            await self.connector.aclose()


def pcm16_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2 == 1:
        raise TtsError(TtsErrorKind.PROVIDER, "PCM16 chunk must contain whole samples")
    int_samples = np.frombuffer(data, dtype="<i2")
    return np.ascontiguousarray(int_samples.astype(np.float32) / 32_768.0)


def provider_from_config(config: ElevenLabsConfig, monotonic_seconds: Clock) -> ElevenLabsTTSProvider:
    return ElevenLabsTTSProvider(
        config=config,
        connector=DefaultHttpConnector(timeout_seconds=config.timeout_seconds),
        monotonic_seconds=monotonic_seconds,
    )


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
