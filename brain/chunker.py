"""Split streaming text into speakable clauses.

The TTS stage wants short units, not sentences, so audio can start while the
brain is still generating. Markers that only make sense on screen (markdown,
code fences, emoji) are stripped before synthesis.
"""

from __future__ import annotations

import re

FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
INLINE_CODE = re.compile(r"`([^`]*)`")
IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_MD = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<!\*)\*([^*\s][^*]*)\*(?!\*)")
EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002b00-\U00002bff"
    "←-⇿"
    "⌀-⏿"
    "]+"
)
BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
HEADING = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
BREATH = re.compile(r"\s*(?:\*\*|<break>|\[(?:pause|breath)\])\s*", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

CLAUSE_END = ".!?…"
HARD_BREAK = re.compile(r"(?<=[.!?…])\s+(?=\S)|(?<=[,;])\s+(?=\S)")
SOFT_BREAK = re.compile(
    r"\s+(?=(?:dan|tapi|tapi\s+saya|namun|sehingga|karena|lalu|jadi|kemudian)\b)",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    text = FENCE.sub(" ", text)
    text = IMAGE_MD.sub(r"\1", text)
    text = LINK_MD.sub(r"\1", text)
    text = BULLET.sub("", text)
    text = HEADING.sub("", text)
    text = INLINE_CODE.sub(r"\1", text)
    text = BOLD.sub(r"\1", text)
    text = ITALIC.sub(r"\1", text)
    text = BREATH.sub(", ", text)
    text = EMOJI.sub("", text)
    return WHITESPACE.sub(" ", text).strip()


def split_prefix(text: str, min_len: int, max_len: int) -> tuple[str, str]:
    """Return the longest leading clause and the remainder."""
    if not text:
        return "", ""
    if len(text) <= min_len:
        return "", text

    if text[-1] in CLAUSE_END:
        return text, ""

    limit = min(max_len, len(text))
    window = text[:limit]

    best = -1
    for pattern in (HARD_BREAK, SOFT_BREAK):
        for match in pattern.finditer(window):
            if match.end() >= min_len:
                best = match.end()
                break
        if best != -1:
            break

    if best == -1:
        if len(text) > max_len:
            cut = window.rfind(" ", min_len, max_len)
            return (window[:cut].strip(), text[cut:].lstrip()) if cut > min_len else (window, text[limit:])
        return "", text

    return text[:best].strip(), text[best:].lstrip()


def clauses(text: str, min_len: int = 18, max_len: int = 140) -> list[str]:
    out: list[str] = []
    rest = text
    while rest:
        head, rest = split_prefix(rest, min_len, max_len)
        if head:
            out.append(head)
        else:
            break
    if rest:
        if out:
            out[-1] = f"{out[-1]} {rest}".strip()
        else:
            out.append(rest)
    return out


class ClauseStreamer:
    """Accumulate deltas and emit whole clauses as soon as they are safe."""

    def __init__(self, min_len: int = 18, max_len: int = 140) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self._buf = ""

    def push(self, text: str) -> list[str]:
        self._buf = clean(self._buf + text)
        out: list[str] = []
        while True:
            head, rest = split_prefix(self._buf, self.min_len, self.max_len)
            if not head:
                break
            out.append(head)
            self._buf = rest
        return out

    def flush(self) -> list[str]:
        self._buf = clean(self._buf)
        if not self._buf:
            return []
        tail = [self._buf]
        self._buf = ""
        return tail

    @property
    def pending(self) -> str:
        return self._buf