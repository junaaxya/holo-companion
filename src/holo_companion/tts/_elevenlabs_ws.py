import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets

from holo_companion.tts.base import TtsError, TtsErrorKind, TtsProviderDiagnostic


class ElevenLabsVoiceRegistration(TypedDict):
    voices: list[str]
    voice_settings: NotRequired[Mapping[str, float]]


class ElevenLabsInput(TypedDict):
    text: str
    voice_id: str


class ElevenLabsInputFrame(TypedDict):
    inputs: list[ElevenLabsInput]
    flush: bool


class ElevenLabsCloseSocketFrame(TypedDict):
    close_socket: bool


class ElevenLabsWebSocket(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def aclose(self) -> None: ...


class ElevenLabsWebSocketConnector(Protocol):
    async def connect(self, url: str, headers: Mapping[str, str]) -> ElevenLabsWebSocket: ...


class WebsocketsNativeSocket(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WebsocketsSocket:
    socket: WebsocketsNativeSocket

    async def send(self, message: str) -> None:
        await self.socket.send(message)

    async def recv(self) -> str | bytes:
        return await self.socket.recv()

    async def aclose(self) -> None:
        await self.socket.close()


@dataclass(frozen=True, slots=True)
class WebsocketsConnector:
    async def connect(self, url: str, headers: Mapping[str, str]) -> ElevenLabsWebSocket:
        socket = await websockets.connect(url, additional_headers=headers)
        return WebsocketsSocket(socket=socket)


@dataclass(frozen=True, slots=True)
class ElevenLabsWsEvent:
    audio: str | None = None
    is_final_audio_for_turn: bool = False
    is_final: bool = False
    error: str | None = None
    error_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None


def text_to_dialogue_url(base_url: str, model_id: str) -> str:
    parsed = urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path}/v1/text-to-dialogue/stream-input" if parsed.path != "" else "/v1/text-to-dialogue/stream-input"
    query = urlencode({"model_id": model_id, "output_format": "pcm_24000", "language_code": "id"})
    return urlunsplit((scheme, parsed.netloc, path, query, ""))


def voice_registration(voice_id: str, voice_settings: Mapping[str, float]) -> ElevenLabsVoiceRegistration:
    registration: ElevenLabsVoiceRegistration = {"voices": [voice_id]}
    stability = voice_settings.get("stability")
    if stability is not None:
        registration["voice_settings"] = {"stability": stability}
    return registration


def input_frame(text: str, voice_id: str) -> ElevenLabsInputFrame:
    return {"inputs": [{"text": text, "voice_id": voice_id}], "flush": True}


def close_socket_frame() -> ElevenLabsCloseSocketFrame:
    return {"close_socket": True}


def parse_ws_event(raw_message: str | bytes) -> ElevenLabsWsEvent:
    try:
        text = raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TtsError(TtsErrorKind.PROVIDER, "ElevenLabs TTS WebSocket emitted invalid JSON") from error
    match payload:
        case {"error": str(error), "code": int(code), "message": str(message)}:
            return ElevenLabsWsEvent(error="provider_error_frame", error_code=code, error_type=error, error_message=sanitize_provider_message(message))
        case {"error": str(error)}:
            return ElevenLabsWsEvent(error="provider_error_frame", error_type=error)
        case {"audio": str(audio)}:
            return ElevenLabsWsEvent(audio=audio, is_final_audio_for_turn=bool(payload.get("is_final_audio_for_turn")), is_final=bool(payload.get("is_final")))
        case {"is_final_audio_for_turn": True}:
            return ElevenLabsWsEvent(is_final_audio_for_turn=True, is_final=bool(payload.get("is_final")))
        case {"is_final": True}:
            return ElevenLabsWsEvent(is_final=True)
        case _:
            raise provider_error("provider_frame", "unsupported_frame", "protocol")


def decode_audio(encoded_audio: str) -> bytes:
    try:
        return base64.b64decode(encoded_audio, validate=True)
    except binascii.Error as error:
        raise provider_error("audio_decode", "invalid_base64", "audio") from error


def provider_error(
    stage: str,
    error: str,
    error_class: str,
    *,
    model: str | None = None,
    status_code: int | None = None,
    close_code: int | None = None,
    message: str | None = None,
) -> TtsError:
    return TtsError(
        TtsErrorKind.PROVIDER,
        "ElevenLabs TTS provider failure",
        TtsProviderDiagnostic(stage, status_code, close_code, error, error_class, message, model),
    )


def sanitize_provider_message(message: str, secrets: tuple[str, ...] = ()) -> str:
    normalized = " ".join(message.split())
    for secret in secrets:
        if secret != "":
            normalized = normalized.replace(secret, "[redacted]")
    return normalized[:160]
