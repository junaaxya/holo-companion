import anyio
import pytest

from holo_companion.runtime.greeting import (
    CURATED_GREETINGS,
    DEFAULT_GREETING_STYLE,
    select_startup_greeting,
)
from holo_companion.runtime.live_cli import _play_startup_greeting
from holo_companion.runtime.playback import PlaybackAdmission
from holo_companion.tts.base import TTSAudioChunk, TtsAudioFormat
from runtime_fakes import FakePlayback, FakeTts


def test_select_startup_greeting_chooses_from_curated_list_and_sets_shy_style() -> None:
    # Given / When
    request = select_startup_greeting()

    # Then
    assert request.text in CURATED_GREETINGS
    assert request.style == DEFAULT_GREETING_STYLE
    assert request.style.emotion == "shy"
    assert request.style.intensity == 0.5
    assert "Master" in request.text
    assert len(CURATED_GREETINGS) >= 4


def test_select_startup_greeting_deterministic_with_custom_choice() -> None:
    # Given
    chosen_text = "Halo, Master tercinta!"

    # When
    request = select_startup_greeting(choices=(chosen_text,), random_fn=lambda seq: seq[0])

    # Then
    assert request.text == chosen_text
    assert request.style.emotion == "shy"


@pytest.mark.anyio
async def test_play_startup_greeting_streams_to_playback_before_listening() -> None:
    # Given
    chunk = TTSAudioChunk.from_samples(
        samples=__import__("numpy").zeros(240, dtype=__import__("numpy").float32),
        audio_format=TtsAudioFormat(sample_rate_hz=24_000),
        sequence=0,
        start_seconds=0.0,
    )
    fake_tts = FakeTts(chunk=chunk)
    fake_playback = FakePlayback()
    out = __import__("io").StringIO()

    # When
    await _play_startup_greeting(fake_tts, fake_playback, out)

    # Then
    assert len(fake_tts.requests) == 1
    assert "Master" in fake_tts.requests[0].text
    assert fake_tts.requests[0].style.emotion == "shy"
    assert len(fake_playback.admissions) == 1
    assert fake_playback.admissions[0].turn_id == 0
    val = out.getvalue()
    assert "greeting:" in val
    assert "startup original text:" in val
    assert "startup text after EmotionMapper:" in val
    assert "exact text submitted to ElevenLabs:" in val
    assert "startup TTS text:" in val
    assert "provider audio chunk count: 1" in val
    assert "playback drain completed: True" in val


@pytest.mark.anyio
async def test_play_startup_greeting_handles_quota_exceeded_gracefully() -> None:
    # Given
    from holo_companion.tts.base import TtsError, TtsErrorKind, TtsProviderDiagnostic

    quota_error = TtsError(
        TtsErrorKind.PROVIDER,
        "ElevenLabs quota exhausted",
        TtsProviderDiagnostic(stage="http_request", error="quota_exceeded", status_code=401),
    )
    fake_tts = FakeTts(chunk=TTSAudioChunk.from_samples(__import__("numpy").zeros(24, dtype=__import__("numpy").float32), TtsAudioFormat(24000), 0, 0.0), fail=True)
    fake_tts.fail = False

    class QuotaTts(FakeTts):
        async def stream(self, request, cancellation=None):
            self.requests.append(request)
            raise quota_error
            yield self.chunk

    tts = QuotaTts(chunk=fake_tts.chunk)
    fake_playback = FakePlayback()
    out = __import__("io").StringIO()

    # When
    await _play_startup_greeting(tts, fake_playback, out)

    # Then
    val = out.getvalue()
    assert len(tts.requests) == 1
    assert "greeting error: ElevenLabs quota exhausted" in val
    assert "playback drain completed: False" in val
    assert "stop/cancel/interruption reason: ElevenLabs quota exhausted" in val
    assert len(fake_playback.admissions) == 0


@pytest.mark.anyio
async def test_startup_greeting_exact_text_consumes_all_generated_pcm() -> None:
    # Given
    text = "Eh, Master... udah balik ya? Aku... kangen pengen ngobrol, hehe."
    chunk1 = TTSAudioChunk.from_samples(
        samples=__import__("numpy").ones(1200, dtype=__import__("numpy").float32),
        audio_format=TtsAudioFormat(sample_rate_hz=24_000),
        sequence=0,
        start_seconds=0.0,
    )
    chunk2 = TTSAudioChunk.from_samples(
        samples=__import__("numpy").ones(2400, dtype=__import__("numpy").float32),
        audio_format=TtsAudioFormat(sample_rate_hz=24_000),
        sequence=1,
        start_seconds=0.05,
    )
    
    class MultiChunkTts(FakeTts):
        async def stream(self, request, cancellation=None):
            self.requests.append(request)
            yield chunk1
            yield chunk2
    
    consumed_count = 0
    fake_tts = MultiChunkTts(chunk=chunk1)
    fake_playback = FakePlayback()
    
    # When
    request = select_startup_greeting(choices=(text,), random_fn=lambda seq: seq[0])
    async for event in fake_tts.stream(request):
        if isinstance(event, TTSAudioChunk):
            await fake_playback.write_and_start(PlaybackAdmission(0, 0, event))
            consumed_count += event.samples.size
            
    await fake_playback.wait_until_drained()

    # Then
    assert len(fake_tts.requests) == 1
    assert fake_tts.requests[0].text == text
    assert len(fake_playback.admissions) == 2
    assert consumed_count == 3600
    assert fake_playback.admissions[0].chunk.samples.size == 1200
    assert fake_playback.admissions[1].chunk.samples.size == 2400


def test_shy_greeting_receives_emotional_delivery_cues() -> None:
    # Given
    request = select_startup_greeting()

    # When / Then
    assert "[shyly]" in request.text
    assert "[softly]" in request.text
    assert "Master" in request.text


def test_normal_neutral_tts_does_not_receive_shy_tags() -> None:
    # Given
    from holo_companion.tts._elevenlabs_emotion import EmotionMapper
    from holo_companion.tts.base import StyleHints

    mapper = EmotionMapper()

    # When
    mapped = mapper.map_text("Halo, Master. Ada yang bisa kubantu?", StyleHints(emotion="neutral"))

    # Then
    assert "[" not in mapped.text
    assert mapped.text == "Halo, Master. Ada yang bisa kubantu?"


def test_audio_tags_survive_normalization() -> None:
    # Given
    from holo_companion.runtime.spoken_text import normalize_spoken_text

    text = "[shyly] H-halo, Master... [softly] akhirnya kamu datang juga."

    # When
    normalized = normalize_spoken_text(text)

    # Then
    assert normalized == text
    assert "[shyly]" in normalized
    assert "[softly]" in normalized


@pytest.mark.anyio
async def test_debug_tts_audio_mode_saves_provider_and_playback_wav(monkeypatch) -> None:
    # Given
    import os
    from pathlib import Path
    monkeypatch.setenv("HOLO_DEBUG_TTS_AUDIO", "1")
    samples = __import__("numpy").ones(2400, dtype=__import__("numpy").float32) * 0.1
    chunk = TTSAudioChunk.from_samples(
        samples=samples,
        audio_format=TtsAudioFormat(sample_rate_hz=24_000),
        sequence=0,
        start_seconds=0.0,
    )
    fake_tts = FakeTts(chunk=chunk)
    fake_playback = FakePlayback()
    out = __import__("io").StringIO()

    # When
    await _play_startup_greeting(fake_tts, fake_playback, out)

    # Then
    val = out.getvalue()
    assert "provider audio chunk count: 1" in val
    assert "provider total samples: 2400" in val
    assert "provider stream completed: True" in val
    assert "playback queued samples: 2400" in val
    assert "playback drain completed: True" in val
    assert "stop/cancel/interruption reason: none" in val
    assert Path("/tmp/holo-provider.wav").exists()
    assert Path("/tmp/holo-playback.wav").exists()


