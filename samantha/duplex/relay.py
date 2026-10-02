"""Relay-race turn: LLM token stream feeds clause-level TTS feeding a live sink.

Unlike :meth:`Samantha.turn`, which waits for the whole reply before synthesising, this
sends text to the TTS while the LLM is still writing. Nothing waits for a stage to complete:
capture, LLM, TTS and playback overlap.

Two details decide the latency, both measured on a POCO F6:

* the opening fragment is cut after 5 words, not at the first sentence boundary, so the
  first clause is synthesised about 1.4s earlier;
* TTS gets 4 of the 8 cores, leaving the rest to the local LLM. At 6 threads the TTS RTF
  degrades from 0.57 to 1.00 and first audio lands at 1.4s instead of 0.7s.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue

from brain.client import BrainBusyError, BrainTransportError, CancelToken
from brain.persona import SYSTEM, sanitize
from samantha.duplex.player import StreamPlayer
from samantha.duplex.profiler import Profiler
from samantha.duplex.streaming_tts import StreamingVoice
from samantha.engine import SR, VOICE_STEPS, Voice, log, sentences_for_tts

MAX_HISTORY_TURNS = 2
MAX_MESSAGE_WORDS = 24


def _trim_context(messages: list[dict]) -> list[dict]:
    """Fold prior turns into the system prompt instead of sending them as messages.

    Measured on Gemma3-1B-IT behind AI Edge Gallery: 13 words of context gave a
    0.26 s time-to-first-token, but adding three history messages pushed it to 11.1 s
    while the token rate stayed identical, so the cost was all in prefill of the
    extra turns. Sending only a compact recap line keeps the prefill tiny.
    """
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not rest:
        return system

    current = rest[-1]
    prior = rest[:-1]
    pairs: list[tuple[dict, dict]] = []
    i = len(prior) - 1
    while i > 0 and len(pairs) < MAX_HISTORY_TURNS:
        reply, user_msg = prior[i], prior[i - 1]
        if reply.get("role") == "assistant" and user_msg.get("role") == "user":
            pairs.append((user_msg, reply))
            i -= 2
        else:
            i -= 1
    recap: list[str] = []
    for user_msg, reply in reversed(pairs):
        asked = " ".join(str(user_msg.get("content", "")).split()[:MAX_MESSAGE_WORDS])
        said = " ".join(str(reply.get("content", "")).split()[:MAX_MESSAGE_WORDS])
        recap.append(f'They asked "{asked}" and you said "{said}".')
    text = str(current.get("content", ""))

    if recap:
        note = "Earlier in this conversation: " + " ".join(recap)
        if system:
            head = str(system[0].get("content", ""))
            system = [{"role": "system", "content": f"{head}\n\n{note}"}]
        else:
            system = [{"role": "system", "content": note}]
    return system + [{"role": "user", "content": text}]


_SOFT_BREAK = {"and", "but", "or", "so", "because", "while", "though", "if", "when", "since"}
_AUX = {"is", "are", "was", "were", "be", "been", "am", "be", "can", "could",
        "will", "would", "should", "may", "might", "must", "do", "does", "did", "has", "have"}


def _first_clause(text: str, head_words: int) -> tuple[str, str] | None:
    """Pick the opening cut point.

    Piper infers a full sentence contour from whatever fragment it is handed, so cutting
    mid-clause makes it sound like the thought ended there. Preference order is therefore:
    the first comma once enough words exist, else a cut just before a conjunction, and
    only as a last resort the raw ``head_words`` cut.
    """
    words = text.split()
    if len(words) < head_words:
        return None
    comma = text.find(",")
    if comma != -1:
        comma_words = len(text[:comma].split())
        if comma_words >= 2 and comma_words + 2 < len(words):
            return text[:comma].strip(), text[comma + 1 :].strip()

    for i in range(head_words, min(len(words), head_words + 4)):
        head_word = words[i].lower().strip(",;:")
        if head_word in _SOFT_BREAK and words[i - 1].lower().strip(",;:") not in _AUX:
            return " ".join(words[:i]), " ".join(words[i:])
    return " ".join(words[:head_words]).rstrip(",;:"), " ".join(words[head_words:])


class RelayTurn:
    """One streaming turn with cancellation shared across all stages."""

    def __init__(
        self,
        voice: Voice,
        brain,
        base_url: str = "http://127.0.0.1:8080",
        tts_threads: int = 4,
        head_words: int = 3,
        speed: float = 0.80,
        latency_msec: int = 120,
        prof: Profiler | None = None,
    ) -> None:
        self.voice = voice
        self.brain = brain
        self.base_url = base_url
        self.tts = StreamingVoice(threads=tts_threads, speed=speed, prof=prof)
        self.head_words = head_words
        self.latency_msec = latency_msec
        self.prof = prof or Profiler("relay-turn")
        self.cancel = CancelToken()
        self._lock = threading.Lock()

    def _first_clause(self, text: str, min_words: int = 3) -> str:
        """Split off the opening clause so audio can start before the full sentence."""
        parts = sentences_for_tts(text)
        if not parts:
            return text.strip()
        head = parts[0]
        if len(parts) > 1 or len(head.split()) >= min_words:
            return head
        return head

    def run(
        self,
        heard: str,
        seconds: int = 5,
        player: StreamPlayer | None = None,
    ) -> dict | None:
        prof = self.prof
        prof.point("transcript ready")
        history = _trim_context(
            list(self.voice.turns)
            + [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": heard},
            ]
        )

        owns_player = player is None
        if owns_player:
            player = StreamPlayer(latency_msec=self.latency_msec)
        t_start = time.monotonic()

        clauses: list[str] = []
        pending = ""
        opened = False
        first_push_at: list[float] = []
        llm_done = threading.Event()
        error: list[str] = []

        def tts_worker() -> None:
            """Consume sentences from the queue as the LLM produces them."""
            while True:
                try:
                    item = q.get(timeout=0.1)
                except Empty:
                    if llm_done.is_set() and q.empty():
                        return
                    continue
                if item is None:
                    return
                text, final = item
                if self.cancel.cancelled:
                    return
                try:
                    self.tts.speak_streaming(text, player)
                    if not first_push_at:
                        first_push_at.append(time.monotonic())
                        prof.point(
                            f"FIRST AUDIO (+{first_push_at[0] - t_start:.3f}s) after clause {text!r}"
                        )
                except Exception as exc:  # noqa: BLE001
                    error.append(f"tts: {type(exc).__name__}: {exc}")
                    return

        q: "Queue[tuple[str, bool] | None]" = Queue()

        worker = threading.Thread(target=tts_worker, daemon=True)
        worker.start()
        prof.point("tts worker started")

        def emit(text: str, final: bool) -> None:
            if not text.strip():
                return
            clauses.append(text.strip())
            q.put((text.strip(), final))

        try:
            for delta in self.brain.stream(history, max_tokens=90, cancel=self.cancel):
                if self.cancel.cancelled:
                    break
                pending += delta
                cut = self._split_ready(pending, head_words=self.head_words, allow_short=opened is False)
                if cut is not None:
                    ready, pending = cut
                    opened = True
                    emit(ready, False)
            if pending.strip() and not self.cancel.cancelled:
                emit(pending.strip(), True)
        except (BrainBusyError, BrainTransportError, TimeoutError, OSError) as exc:
            error.append(f"llm: {type(exc).__name__}: {exc}")
        finally:
            llm_done.set()
            q.put(None)

        worker.join(timeout=120)
        prof.point("all clauses synthesised")

        if owns_player:
            player.close()
            prof.point("playback drained")

        reply = sanitize(" ".join(clauses))
        first_audio = (first_push_at[0] - t_start) if first_push_at else None
        self.voice.turns.extend(
            [{"role": "user", "content": heard}, {"role": "assistant", "content": reply}]
        )
        log(f"RELAY first audio: {first_audio if first_audio else 'n/a'}s | {reply[:80]!r}")
        if error:
            log(f"relay errors: {error}")
        return {
            "heard": heard,
            "reply": reply,
            "clauses": clauses,
            "first_audio_s": first_audio,
            "errors": error,
            "total_s": prof.elapsed,
            "prof": prof,
        }

    @staticmethod
    def _split_ready(text: str, head_words: int = 5, allow_short: bool = True):
        """Return ``(ready_text, rest)`` for the next TTS unit.

        ``allow_short`` only applies to the opening fragment: that one is cut after
        ``head_words`` words so audio can start while the LLM is still writing, because
        that is what dominates time-to-first-audio. Every later unit waits for a real
        sentence boundary, otherwise the reply turns into a stutter of two-word chunks
        and the prosody falls apart.
        """
        stripped = text.strip()
        if not stripped:
            return None
        if allow_short:
            cut = _first_clause(stripped, head_words)
            if cut is not None:
                return cut
        for i in range(len(stripped) - 1, -1, -1):
            if stripped[i] in ".!?":
                head = stripped[: i + 1].strip()
                rest = stripped[i + 1 :].strip()
                if head and rest:
                    return head, rest
        return None