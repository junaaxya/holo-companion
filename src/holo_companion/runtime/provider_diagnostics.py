from holo_companion.llm.base import LlmError
from holo_companion.runtime.turn import ProviderDiagnostic
from holo_companion.stt.base import SttError
from holo_companion.tts.base import TtsError


def provider_diagnostic(error: SttError | LlmError | TtsError) -> ProviderDiagnostic:
    match error:
        case LlmError(kind=kind, status_code=status_code):
            return ProviderDiagnostic("openai-compatible", type(error).__name__, kind.value, status_code, kind.value)
        case TtsError(kind=kind, diagnostic=diagnostic):
            message = "ElevenLabs quota exhausted" if diagnostic is not None and diagnostic.error == "quota_exceeded" else kind.value
            return ProviderDiagnostic(
                "elevenlabs",
                type(error).__name__,
                message,
                None if diagnostic is None else diagnostic.status_code,
                None if diagnostic is None else diagnostic.error,
            )
        case SttError():
            return ProviderDiagnostic("faster-whisper", type(error).__name__, "stt provider failure")
        case unreachable:
            raise AssertionError(f"unexpected provider error: {unreachable}")
