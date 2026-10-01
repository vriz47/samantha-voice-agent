"""Contract tests for BrainClient against a live LiteRT-LM Gallery server.

The server is single-slot and crashes on aborted streams, so these tests
deliberately check drain-to-completion and never cancel.
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brain.client import BrainClient, CancelToken

URL = os.environ.get("SAMANTHA_BRAIN_URL", "http://127.0.0.1:8080")


def test_health_and_single_turn():
    c = BrainClient(URL)
    assert c.healthy(), "server tidak merespons /health"
    t0 = time.time()
    out = c.ask(
        [
            {"role": "system", "content": "Kamu Samantha. Jawab Bahasa Indonesia, satu kalimat."},
            {"role": "user", "content": "Sebut satu warna."},
        ],
        max_tokens=40,
    )
    assert out.strip(), "respons kosong"
    print(f"\n  turn: {time.time() - t0:.2f}s -> {out.strip()[:80]}")


def test_abandoning_generator_does_not_kill_server():
    c = BrainClient(URL)
    gen = c.stream([{"role": "user", "content": "Tulis paragraf panjang tentang hujan."}], max_tokens=400)
    first = next(gen)
    assert first
    del gen
    time.sleep(3.0)
    assert c.wait_ready(30), "server mati setelah generator dibuang"


def test_concurrent_calls_are_serialised_not_rejected():
    c = BrainClient(URL)
    results: dict[int, str] = {}
    errors: dict[int, str] = {}

    def run(i: int) -> None:
        try:
            results[i] = c.ask([{"role": "user", "content": f"Sebut angka {i}."}], max_tokens=30)
        except Exception as exc:
            errors[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    print(f"\n  3 concurrent in {wall:.2f}s, ok={len(results)} err={len(errors)}")
    for i, e in errors.items():
        print("   err", i, e)
    assert len(results) == 3, "client serialiser gagal mencegah 429"
    assert c.healthy()


def test_cancel_discards_output_but_server_survives():
    c = BrainClient(URL)
    token = CancelToken()
    gen = c.stream(
        [{"role": "user", "content": "Tulis cerita panjang tentang gunung."}],
        max_tokens=400,
        cancel=token,
    )
    seen = []
    for piece in gen:
        seen.append(piece)
        if len(seen) == 3:
            token.cancel()
    assert len(seen) >= 3
    time.sleep(3.0)
    assert c.wait_ready(30), "server mati setelah cancel-by-discard"
    print("\n cancel: output dibuang, server tetap hidup")


def test_clause_streaming_keeps_up_with_generator():
    from brain.chunker import ClauseStreamer

    c = BrainClient(URL)
    s = ClauseStreamer(min_len=14, max_len=90)
    t0 = time.time()
    first_clause_at = None
    text = ""
    for delta in c.stream(
        [{"role": "system", "content": "Kamu Samantha. Jawab Bahasa Indonesia."},
         {"role": "user", "content": "Ceritakan rencanamu hari ini, agak panjang."}],
        max_tokens=160,
    ):
        text += delta
        got = s.push(delta)
        if got and first_clause_at is None:
            first_clause_at = time.time() - t0
            print(f"\n  first clause at {first_clause_at:.2f}s -> {got[0][:60]}")
    tail = s.flush()
    print(f"  total {time.time() - t0:.2f}s, {len(tail)} tail clause(s)")
    assert first_clause_at is not None, "tidak ada clause yang keluar selama streaming"
    assert (time.time() - t0) - first_clause_at < 5.0, "clause pertama keluar terlalu telat"
    assert text.strip()