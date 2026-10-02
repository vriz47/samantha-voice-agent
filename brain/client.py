"""Samantha brain client for LiteRT-LM Gallery server.

Safety contract learned from probing com.server.edge.gallery (LiteRT-LM v27):
the Ktor/Netty backend crashes with ClosedWriteChannelException when a client
closes the socket mid-stream. A cancelled request must therefore always be
drained to completion server-side; only the client-side output is discarded.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from queue import Queue
from typing import Iterator, Sequence

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = os.environ.get("SAMANTHA_MODEL", "Gemma-4-E2B-it")


class BrainBusyError(RuntimeError):
    """Server rejected every retry attempt because the model slot was busy."""


class BrainTransportError(RuntimeError):
    """Transport-level failure that was not a busy rejection."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class TurnStats:
    ttft: float | None = None
    total: float = 0.0
    chunks: int = 0
    chars: int = 0
    attempts: int = 1
    cancelled: bool = False
    text: str = field(default="", repr=False)


def _parse_sse(raw: str) -> str | None:
    payload = raw[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


class BrainClient:
    """Single-slot client. Serialises requests so the server never sees a 429."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        max_attempts: int = 6,
        backoff_base: float = 0.4,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8080
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._slot = threading.Lock()

    def healthy(self) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status == 200
        except OSError:
            return False

    def wait_ready(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.healthy():
                return True
            time.sleep(1.0)
        return False

    def stream(
        self,
        messages: Sequence[dict],
        max_tokens: int = 90,
        temperature: float | None = 0.6,
        top_p: float | None = 0.9,
        cancel: CancelToken | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for one turn.

        Abandoning the generator early is safe: the socket is always drained to
        [DONE] by the worker thread so the server keeps running.
        """
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        blob = json.dumps(payload).encode()

        deltas: Queue[str | None] = Queue()
        result = TurnStats()

        def worker() -> None:
            try:
                self._run(blob, deltas, result, cancel)
            except BaseException as exc:  # surfaced through the queue
                deltas.put(exc)
            finally:
                self._slot.release()
                deltas.put(None)

        self._slot.acquire()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = deltas.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            if item == "":
                continue
            if cancel is not None and cancel.cancelled:
                result.cancelled = True
                continue
            result.chunks += 1
            result.chars += len(item)
            result.text += item
            yield item

    def _run(
        self,
        blob: bytes,
        deltas: "Queue[str | None]",
        result: TurnStats,
        cancel: CancelToken | None,
    ) -> None:
        payload = json.loads(blob.decode("utf-8"))
        headers = {"Content-Type": "application/json"}
        started = time.time()
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                conn.request("POST", "/v1/chat/completions", body=blob, headers=headers)
                resp = conn.getresponse()

                if resp.status == 429:
                    resp.read()
                    last_error = BrainBusyError("model slot busy")
                    self._sleep_backoff(attempt)
                    continue

                if resp.status != 200:
                    detail = resp.read()[:200].decode("utf-8", "replace")
                    if resp.status == 404 and self._adopt_served_model():
                        blob = json.dumps({**payload, "model": self.model}).encode()
                    last_error = BrainTransportError(f"HTTP {resp.status}: {detail}")
                    self._sleep_backoff(attempt)
                    continue

                first = True
                buf = b""
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        text = _parse_sse(line)
                        if text is None:
                            result.total = time.time() - started
                            return
                        if text:
                            if first:
                                result.ttft = time.time() - started
                                first = False
                            deltas.put(text)
                result.total = time.time() - started
                return
            except (OSError, http.client.HTTPException) as exc:
                last_error = BrainTransportError(f"{type(exc).__name__}: {exc}")
                self._sleep_backoff(attempt)
            finally:
                conn.close()

        if isinstance(last_error, BrainBusyError):
            raise BrainBusyError(f"still busy after {self.max_attempts} attempts") from last_error
        raise last_error or BrainTransportError("unknown failure")

    def _adopt_served_model(self) -> bool:
        """Point the client at the model the server actually loaded.

        The Gallery app only ever holds one model in its single slot, and switching
        models in the UI changes it under us. A 404 means our requested model id is gone,
        so read /v1/models and retry with whatever is live instead of failing the turn.
        """
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8", "replace"))
            conn.close()
        except (OSError, http.client.HTTPException, ValueError):
            return False
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        if len(ids) == 1 and ids[0] != self.model:
            self.model = ids[0]
            return True
        return False

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= self.max_attempts:
            return
        delay = min(self.backoff_base * (2 ** (attempt - 1)), 5.0)
        time.sleep(delay + random.uniform(0, 0.15))

    def ask(
        self,
        messages: Sequence[dict],
        max_tokens: int = 90,
        temperature: float | None = 0.7,
        cancel: CancelToken | None = None,
    ) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens, temperature=temperature, cancel=cancel))