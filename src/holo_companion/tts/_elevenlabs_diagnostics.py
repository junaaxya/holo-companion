from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException

from holo_companion.tts.base import TtsError, TtsErrorKind, TtsProviderDiagnostic
from holo_companion.tts._elevenlabs_ws import sanitize_provider_message


def config_error(message: str, model: str) -> TtsError:
    return TtsError(
        TtsErrorKind.CONFIG,
        message,
        TtsProviderDiagnostic("config", error="invalid_config", error_class="config", message=message, model=model),
    )


def provider_diagnostic_error(
    stage: str,
    error: str,
    error_class: str,
    model: str,
    *,
    status_code: int | None = None,
    close_code: int | None = None,
) -> TtsError:
    return TtsError(
        TtsErrorKind.PROVIDER,
        "ElevenLabs TTS provider failure",
        TtsProviderDiagnostic(stage, status_code, close_code, error, error_class, None, model),
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
