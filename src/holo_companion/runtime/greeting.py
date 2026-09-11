import random
from collections.abc import Sequence

from holo_companion.tts.base import StyleHints, SynthesisRequest

CURATED_GREETINGS: tuple[str, ...] = (
    "[shyly] H-halo, Master... [softly] akhirnya kamu datang juga. Hehe... aku udah nungguin.",
    "[shyly] Eh, Master... [softly] udah balik ya? Aku... kangen pengen ngobrol, hehe.",
    "[shyly] H-hai, Master... [softly] Masuk aja, aku udah siap kok buat nemenin kamu.",
    "[shyly] Master... [softly] akhirnya kamu luang juga. Jangan capek-capek ya, yuk ngobrol sebentar.",
)

DEFAULT_GREETING_STYLE = StyleHints(emotion="shy", intensity=0.5, speaking_style="soft")


def select_startup_greeting(
    choices: Sequence[str] = CURATED_GREETINGS,
    random_fn=random.choice,
) -> SynthesisRequest:
    text = random_fn(choices)
    return SynthesisRequest(text=text, style=DEFAULT_GREETING_STYLE)
