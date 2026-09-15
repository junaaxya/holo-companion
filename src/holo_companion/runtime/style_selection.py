import re

from holo_companion.tts.base import StyleHints

_ROMANTIC_PATTERN = re.compile(
    r"\b(sayang|cinta|kangen|rindu|jadian|pacar|pacaran|peluk|cium|gebetan|salah tingkah)\b|"
    r"maunya kamu|suka (sama )?kamu|sayang (sama )?kamu|kangen kamu|cinta (sama )?kamu",
    re.IGNORECASE,
)
_PRAISE_PATTERN = re.compile(
    r"\b(cantik|imut|lucu|gemas|gemesin|pintar|pinter|hebat|keren|manis)\b",
    re.IGNORECASE,
)
_SAD_SERIOUS_PATTERN = re.compile(
    r"\b(sedih|nangis|menangis|capek|lelah|penat|stres|stress|galau|sakit|kecewa|putus asa|bingung|berat|masalah|kesepian)\b",
    re.IGNORECASE,
)


def select_contextual_style(text: str) -> StyleHints:
    normalized = text.strip()
    if _ROMANTIC_PATTERN.search(normalized):
        return StyleHints(emotion="shy", intensity=0.5, speaking_style="playful")
    if _PRAISE_PATTERN.search(normalized):
        return StyleHints(emotion="shy", intensity=0.5, speaking_style="soft")
    if _SAD_SERIOUS_PATTERN.search(normalized):
        return StyleHints(emotion="concerned", intensity=0.5, speaking_style="soft")
    return StyleHints(emotion="neutral", energy=0.5, intensity=0.5, speaking_style="neutral")
