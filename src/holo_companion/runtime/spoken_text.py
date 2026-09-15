import re

_MARKDOWN_HEADER = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*")
_MARKDOWN_LIST = re.compile(r"(?m)^[ \t]*(?:[-+*]|\d+[.)])(?=[ \t]|$)[ \t]*")
_INLINE_NUMBERED = re.compile(r"(?:^|\s+)\d+[.)](?=[ \t]|$)\s*")
_HEADING_COLON_LINE = re.compile(r"(?m)^[ \t]*([A-Za-z0-9_]{1,16}:)[ \t]*\n")


def normalize_spoken_text(text: str) -> str:
    cleaned = _MARKDOWN_HEADER.sub("", text)
    cleaned = _MARKDOWN_LIST.sub("", cleaned)
    cleaned = _INLINE_NUMBERED.sub(" ", cleaned)
    cleaned = _HEADING_COLON_LINE.sub(r"\1 ", cleaned)
    cleaned = cleaned.replace("**", "").replace("*", "")
    return cleaned


