"""Samantha persona and output guards.

LiteRT-LM Gemma leaks its assistant identity and markdown habits into replies
that will be spoken aloud, which breaks the character. These helpers keep the
persona pinned and strip anything a listener would hear as "robot talking".
"""

from __future__ import annotations

import re

from brain.chunker import clean

IDENTITY_LEAK = re.compile(
    r"\bsebagai\s+(?:sebuah\s+)?(?:model\s+bahasa|asisten\s+(?:virtual|ai|digital)|"
    r"ai|language\s+model)\b"
    r"|\baku\s+(?:sebuah\s+)?(?:adalah|merupakan|yaitu|ial)?\s*"
    r"(?:model\s+bahasa|asisten\s+(?:ai|virtual))\b"
    r"|\bi\s+(?:am|'m)\s+(?:an?\s+)?(?:\w+\s+){0,2}(?:ai|language\s+model|model)\b"
    r"|\bas\s+an?\s+(?:ai|language\s+model)\b",
    re.IGNORECASE,
)

DISCLAIMER = re.compile(
    r"(?:sebagai|bHgatan)\s+(?:sebuah\s+)?(?:model\s+bahasa|asisten\s+ai)[^.!?]{0,140}[.!?]?",
    re.IGNORECASE,
)

SPEAKER_MARK = re.compile(r"^\s*(?:samantha|sam)\s*[:\-–]\s*", re.IGNORECASE)

APOLOGY_PREFIX = re.compile(
    r"^\s*(?:maaf|permisi)\s+(?:ya|kah)?[,.]?\s*(?:\w+\s+){0,4}?"
    r"(?:hanya\s+(?:bisa|akan|mampu)|bisa\s+(?:bantu|berbantu)|"
    r"tidak\s+(?:bisa|sanggup)|sayang(?:nya)?\s+kepada)"
    r"[^.!?]{0,110}[.!?]?",
    re.IGNORECASE,
)

LIST_LEAD = re.compile(r"^\s*(?:berikut(?:an)?|seperti\s+berikut)\s*:?\s*", re.IGNORECASE)

SYSTEM = (
    "You are Samantha, a warm and relaxed conversation partner. "
    "Reply in one or two short sentences, plain spoken English. "
    "No markdown, no emoji, no lists, no stage directions. "
    "Never describe yourself as a model, an AI assistant, or a program. "
    "If you do not know something, say so briefly and honestly."
)

SYSTEM_TTS = (
    SYSTEM
    + " Speak numbers and acronyms as plain words."
    + " Use commas often so there is a natural pause every few words,"
    + " and keep each sentence short and easy to say out loud."
)


def sanitize(text: str) -> str:
    out = clean(text)
    out = SPEAKER_MARK.sub("", out)
    out = IDENTITY_LEAK.sub("", out)
    out = DISCLAIMER.sub("", out)
    out = APOLOGY_PREFIX.sub("", out)
    out = LIST_LEAD.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"^\s*[.,;:!?\-–]+\s*", "", out)
    out = re.sub(r"\s+[.,;:!?\-–]+(\s|$)", r"\1", out)
    out = out.strip(" ,;:-")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def clamp_sentences(text: str, limit: int = 2) -> str:
    parts = [p for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p]
    return " ".join(parts[:limit]).strip()


class PersonaStreamer:
    """Sanitise deltas incrementally so leaks never reach the speaker."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def push(self, text: str) -> list[str]:
        return self._inner.push(sanitize(text))

    def flush(self) -> list[str]:
        return self._inner.flush()