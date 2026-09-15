import json
from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException

from holo_companion.tts.base import TtsError, TtsErrorKind, TtsProviderDiagnostic
from holo_companion.tts._elevenlabs_ws import sanitize_provider_message


def config_error(message: str, model: str) -> TtsError:
    return TtsError(
        TtsErrorKind.CONFIG,
        message,
        TtsProviderDiagnostic("config", error="invalid_config", error_class="config", message=message, model=model),
    )


def is_quota_exceeded(body: bytes) -> bool:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            detail = data.get("detail")
            if isinstance(detail, dict):
                code = str(detail.get("status") or detail.get("code") or "")
                msg = str(detail.get("message") or "")
                if code == "quota_exceeded" or "quota" in code.lower() or "credits remaining" in msg.lower():
                    return True
            elif isinstance(detail, str) and ("quota_exceeded" in detail.lower() or "quota" in detail.lower()):
                return True
            code = str(data.get("code") or data.get("status") or data.get("error") or "")
            msg = str(data.get("message") or "")
            if code == "quota_exceeded" or "quota" in code.lower() or "credits remaining" in msg.lower():
                return True
    except Exception:
        pass
    text_lower = body.decode("utf-8", errors="replace").lower()
    return "quota_exceeded" in text_lower or "0 credits remaining" in text_lower


def provider_diagnostic_error(
    stage: str,
    error: str,
    error_class: str,
    model: str,
    *,
    status_code: int | None = None,
    close_code: int | None = None,
) -> TtsError:
    message = "ElevenLabs quota exhausted" if error == "quota_exceeded" else None
    error_msg = message or "ElevenLabs TTS provider failure"
    return TtsError(
        TtsErrorKind.PROVIDER,
        error_msg,
        TtsProviderDiagnostic(stage, status_code, close_code, error, error_class, message, model),
    )


def handshake_status_code(error: InvalidHandshake | WebSocketException) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def socket_close_code(error: ConnectionClosed | StopAsyncIteration) -> int | None:
    received = getattr(error, "rcvd", None)
    code = getattr(received, "code", None)
    return code if isinstance(code, int) else None


def with_model(error: TtsError, model: str) -> TtsError:
    diagnostic = error.diagnostic
    if diagnostic is None:
        return provider_diagnostic_error("provider_frame", "protocol_error", "protocol", model)
    return TtsError(
        error.kind,
        error.message,
        TtsProviderDiagnostic(
            diagnostic.stage,
            diagnostic.status_code,
            diagnostic.close_code,
            diagnostic.error,
            diagnostic.error_class,
            diagnostic.message,
            model,
        ),
    )


def sanitize_message(message: str, api_key: str, voice_id: str, text: str) -> str:
    return sanitize_provider_message(message, (api_key, voice_id, text))
