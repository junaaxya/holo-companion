import re
from dataclasses import dataclass
from typing import Final

SENTENCE_BOUNDARIES: Final[frozenset[str]] = frozenset((".", "?", "!", "…", "\n"))
DEFAULT_FALLBACK_CHARS: Final = 160
DEFAULT_MIN_SEGMENT_CHARS: Final = 8

_ORDINAL_OR_SHORT = re.compile(r"^\s*(?:\d+[.)]\s*|[A-Za-z0-9_]{1,12}:?\s*)$", re.DOTALL)


@dataclass(slots=True)
class IncrementalSpeechSegmenter:
    fallback_chars: int = DEFAULT_FALLBACK_CHARS
    min_segment_chars: int = 1
    _buffer: str = ""

    def __post_init__(self) -> None:
        if self.fallback_chars <= 0:
            raise ValueError("fallback_chars must be greater than 0")

    def push(self, delta: str) -> tuple[str, ...]:
        if delta == "":
            return ()
        if self._buffer.endswith(" ") and delta.startswith(" "):
            delta = delta.lstrip(" ")
        self._buffer += delta
        return self._drain_segments()

    def flush(self) -> tuple[str, ...]:
        residual = self._buffer
        self._buffer = ""
        if residual.strip() == "":
            return ()
        return (residual,)

    def _drain_segments(self) -> tuple[str, ...]:
        segments: list[str] = []
        while self._buffer.strip() != "":
            boundary = _first_boundary_index(self._buffer)
            if boundary is not None:
                split_at = boundary + 1
                while split_at < len(self._buffer) and self._buffer[split_at] in {".", "!", "?", "…"}:
                    split_at += 1
                segment = self._buffer[:split_at]
                self._buffer = self._buffer[split_at:]
                stripped = segment.strip()
                if stripped == "" or all(ch in {".", "!", "?", "…"} for ch in stripped):
                    continue
                if _ORDINAL_OR_SHORT.match(stripped) and self._buffer.strip() != "":
                    next_boundary = _first_boundary_index(self._buffer)
                    if next_boundary is None:
                        self._buffer = segment + self._buffer
                        break
                    next_split = next_boundary + 1
                    while next_split < len(self._buffer) and self._buffer[next_split] in {".", "!", "?", "…"}:
                        next_split += 1
                    merged = segment.rstrip() + " " + self._buffer[:next_split].lstrip()
                    self._buffer = self._buffer[next_split:]
                    segments.append(merged)
                    continue
                segments.append(segment)
                continue

            if len(self._buffer) < self.fallback_chars:
                break
            fallback = _last_whitespace_before(self._buffer, self.fallback_chars)
            if fallback is None:
                break
            segments.append(self._buffer[:fallback].rstrip())
            self._buffer = self._buffer[fallback:]
        if self._buffer.strip() == "":
            self._buffer = ""
        return tuple(segments)


def _first_boundary_index(text: str) -> int | None:
    indexes = [text.find(boundary) for boundary in SENTENCE_BOUNDARIES if text.find(boundary) >= 0]
    if len(indexes) == 0:
        return None
    return min(indexes)


def _last_whitespace_before(text: str, threshold: int) -> int | None:
    search_end = min(len(text), threshold + 1)
    for index in range(search_end - 1, -1, -1):
        if text[index].isspace():
            return index
    return None
