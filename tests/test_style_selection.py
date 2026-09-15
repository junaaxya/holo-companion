import pytest

from holo_companion.runtime.style_selection import select_contextual_style


@pytest.mark.parametrize(
    "text",
    [
        "Aku maunya kamu, boleh gak?",
        "Aku kangen kamu.",
        "Sayang, lagi apa?",
        "Cinta banget sama kamu, Master.",
        "Boleh peluk gak?",
        "Bikin salah tingkah aja deh.",
    ],
)
def test_select_contextual_style_romantic_flirty_maps_to_shy_playful(text: str) -> None:
    # Given / When
    style = select_contextual_style(text)

    # Then
    assert style.emotion == "shy"
    assert style.intensity == 0.5
    assert style.speaking_style == "playful"


@pytest.mark.parametrize(
    "text",
    [
        "Kamu imut banget deh.",
        "Holo cantik banget hari ini.",
        "Pinter banget sih kamu.",
        "Hebat banget, makasih ya.",
        "Lucu banget sih.",
        "Manis banget deh kamu.",
    ],
)
def test_select_contextual_style_praise_compliment_maps_to_shy_soft(text: str) -> None:
    # Given / When
    style = select_contextual_style(text)

    # Then
    assert style.emotion == "shy"
    assert style.intensity == 0.5
    assert style.speaking_style == "soft"


@pytest.mark.parametrize(
    "text",
    [
        "Aku capek banget hari ini.",
        "Lagi sedih banget...",
        "Pengen nangis rasanya.",
        "Banyak masalah di kerjaan, stres.",
        "Lelah banget badanku.",
    ],
)
def test_select_contextual_style_serious_sad_maps_to_concerned_soft(text: str) -> None:
    # Given / When
    style = select_contextual_style(text)

    # Then
    assert style.emotion == "concerned"
    assert style.intensity == 0.5
    assert style.speaking_style == "soft"


@pytest.mark.parametrize(
    "text",
    [
        "Minum apa enaknya?",
        "Lagi ngapain sekarang?",
        "Cuaca hari ini cerah ya.",
        "Gimana kabarnya?",
        "Bahas Python yuk.",
    ],
)
def test_select_contextual_style_casual_maps_to_warm_neutral(text: str) -> None:
    # Given / When
    style = select_contextual_style(text)

    # Then
    assert style.emotion == "neutral"
    assert style.speaking_style == "neutral"
