from holo_companion.runtime.segmenter import IncrementalSpeechSegmenter
from holo_companion.runtime.spoken_text import normalize_spoken_text


def test_segmenter_preserves_content_across_arbitrary_deltas() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter(fallback_chars=160)

    # When
    emitted = [segment for delta in ("Ha", "lo", ", Ho", "lo.", " Apa", " kab", "ar?") for segment in segmenter.push(delta)]
    emitted.extend(segmenter.flush())

    # Then
    assert emitted == ["Halo, Holo.", " Apa kabar?"]
    assert "".join(emitted) == "Halo, Holo. Apa kabar?"


def test_segmenter_splits_on_sentence_boundaries_and_newlines() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter(fallback_chars=160)

    # When
    emitted = segmenter.push("Satu! Dua? Tiga…\nEmpat.")

    # Then
    assert emitted == ("Satu!", " Dua?", " Tiga…", "Empat.")


def test_segmenter_flushes_trailing_residual_and_ignores_whitespace_only() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter(fallback_chars=160)

    # When
    whitespace = segmenter.push("   \n\t")
    segmenter.push("Belum selesai")
    residual = segmenter.flush()

    # Then
    assert whitespace == ()
    assert residual == ("Belum selesai",)
    assert segmenter.flush() == ()


def test_segmenter_fallback_splits_at_last_whitespace_without_splitting_word() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter(fallback_chars=160)
    prefix = "kata " * 32
    suffix = "panjang"

    # When
    emitted = segmenter.push(prefix + suffix)
    residual = segmenter.flush()

    # Then
    assert emitted == (prefix.rstrip(),)
    assert residual == (" " + suffix,)
    assert "".join(emitted + residual) == prefix + suffix


def test_segmenter_does_not_split_long_word_at_fallback_threshold() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter(fallback_chars=160)
    long_word = "a" * 200

    # When
    emitted = segmenter.push(long_word)
    residual = segmenter.flush()

    # Then
    assert emitted == ()
    assert residual == (long_word,)


def test_segmenter_merges_short_numbered_fragment_with_following_sentence() -> None:
    # Given
    segmenter = IncrementalSpeechSegmenter()

    # When
    emitted = segmenter.push("1. Seduh kopi dengan air hangat.")

    # Then
    assert emitted == ("1. Seduh kopi dengan air hangat.",)


def test_normalize_spoken_text_removes_markdown_list_markers_and_emphasis() -> None:
    # Given
    raw = "## Makanan:\n1. **Kopi**\n- kacang tanah"

    # When
    normalized = normalize_spoken_text(raw)

    # Then
    assert normalized == "Makanan: Kopi\nkacang tanah"


def test_normalize_spoken_text_preserves_hyphenated_indonesian_words() -> None:
    # Given
    raw = "## Tips:\n- malam-malam jangan lupa jaket\n- jalan-jalan sore\n1. hati-hati di jalan ya"

    # When
    normalized = normalize_spoken_text(raw)

    # Then
    assert "malam-malam" in normalized
    assert "jalan-jalan" in normalized
    assert "hati-hati" in normalized
    assert normalized == "Tips: malam-malam jangan lupa jaket\njalan-jalan sore\nhati-hati di jalan ya"
    assert normalize_spoken_text("-malam") == "-malam"
    assert normalize_spoken_text("malam-malam") == "malam-malam"
