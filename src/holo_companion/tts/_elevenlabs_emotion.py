from dataclasses import dataclass

from holo_companion.tts.base import StyleHints, TtsError, TtsErrorKind


@dataclass(frozen=True, slots=True)
class MappedVoiceText:
    text: str


class EmotionMapper:
    def map_text(self, text: str, style: StyleHints | None) -> MappedVoiceText:
        if style is None or style.emotion is None or style.emotion == "neutral":
            return MappedVoiceText(text=text)
        if text.startswith("["):
            return MappedVoiceText(text=text)
        intensity = style.intensity if style.intensity is not None else style.energy
        band = intensity_band(0.5 if intensity is None else intensity)
        match (style.emotion, band):
            case ("happy", "low"):
                tag, punctuation = "[happily]", "preserve"
            case ("happy", "medium"):
                tag, punctuation = "[happily]", "exclaim"
            case ("happy", "high"):
                tag, punctuation = "[excited]", "exclaim"
            case ("playful", "low"):
                tag, punctuation = "[mischievously]", "preserve"
            case ("playful", "medium" | "high"):
                tag, punctuation = "[mischievously]", "exclaim"
            case ("relaxed", "low" | "medium"):
                tag, punctuation = "[softly]", "preserve"
            case ("relaxed", "high"):
                tag, punctuation = "[softly]", "ellipsis"
            case ("concerned", "low"):
                tag, punctuation = "[reassuring]", "preserve"
            case ("concerned", "medium"):
                tag, punctuation = "[worried]", "preserve"
            case ("concerned", "high"):
                tag, punctuation = "[worried]", "ellipsis"
            case ("sad", "low"):
                tag, punctuation = "[sad]", "preserve"
            case ("sad", "medium" | "high"):
                tag, punctuation = "[sad]", "ellipsis"
            case ("angry", "low"):
                tag, punctuation = "[frustrated]", "preserve"
            case ("angry", "medium"):
                tag, punctuation = "[angry]", "preserve"
            case ("angry", "high"):
                tag, punctuation = "[angry]", "exclaim"
            case ("surprised", "low"):
                tag, punctuation = "[surprised]", "preserve"
            case ("surprised", "medium"):
                tag, punctuation = "[surprised]", "exclaim"
            case ("surprised", "high"):
                tag, punctuation = "[excited]", "exclaim"
            case ("shy", "low"):
                tag, punctuation = "[hesitantly]", "preserve"
            case ("shy", "medium"):
                tag, punctuation = "[hesitantly]", "ellipsis"
            case ("shy", "high"):
                tag, punctuation = "[whispers]", "ellipsis"
            case _:
                raise TtsError(TtsErrorKind.CONFIG, f"unsupported emotion for ElevenLabs TTS: {style.emotion}")
        return MappedVoiceText(text=f"{tag} {with_delivery_punctuation(text, punctuation)}")


def intensity_band(intensity: float) -> str:
    if intensity < 1 / 3:
        return "low"
    return "medium" if intensity < 2 / 3 else "high"


def with_delivery_punctuation(text: str, punctuation: str) -> str:
    if punctuation == "preserve":
        return text
    if punctuation == "exclaim":
        return f"{text.rstrip('.!?')}!"
    if text.endswith(("...", "?", "!")):
        return text
    return f"{text.rstrip('.')}..."
