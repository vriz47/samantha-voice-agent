"""Benchmark whatever model the LiteRT-LM Gallery server currently has loaded.

Usage:  python tools/bench_brain.py [prompt_count]

Auto-detects the active model id from /v1/models, so no code edit is needed after
switching models in the Gallery app.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.client import BrainClient  # noqa: E402
from brain.persona import SYSTEM  # noqa: E402

BASE = "http://127.0.0.1:8080"
PROMPTS = [
    "Tell me about the sea.",
    "What is your favourite season?",
    "Say something about the moon.",
    "How do you feel when it rains?",
    "Explain what a rainbow is.",
    "Who are you?",
]


def active_model() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
        data = json.load(r)
    ids = [m["id"] for m in data.get("data", [])]
    return ids[0] if ids else "?"


def mem() -> tuple[float, float]:
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    avail = avail = 0
    swap = 0
    for line in meminfo:
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / 1024
        elif line.startswith("SwapFree:"):
            swap = int(line.split()[1]) / 1024
    return avail, swap


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
    prompts = PROMPTS[:n]
    model = active_model()
    avail, swapfree = mem()
    print(f"model     : {model}")
    print(f"MemAvail  : {avail:.0f} MB | SwapFree: {swapfree:.0f} MB")
    print()

    brain = BrainClient(model=model)
    ttfts: list[float] = []
    five: list[float] = []
    totals: list[float] = []
    cot = 0

    print(f"{'prompt':32s} {'ttft':>7s} {'5kata':>7s} {'total':>7s}  think  preview")
    for q in prompts:
        t0 = time.monotonic()
        first = None
        at5 = None
        buf = ""
        for delta in brain.stream(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            max_tokens=120,
        ):
            if first is None and delta:
                first = time.monotonic() - t0
            buf += delta
            if at5 is None and len(buf.split()) >= 5:
                at5 = time.monotonic() - t0
        total = time.monotonic() - t0
        has_cot = bool(re.search(r"<think>|</think>|let me (think|figure)|i need to (think|figure)", buf, re.I))
        cot += has_cot
        ttfts.append(first or total)
        five.append(at5 or total)
        totals.append(total)
        print(
            f"{q[:31]:32s} {first or 0:6.3f}s {at5 or 0:6.3f}s {total:6.3f}s"
            f"  {'YA' if has_cot else 'no':5s}  {buf[:40]!r}"
        )

    print()
    print(f"median ttft {statistics.median(ttfts):.3f}s | median 5kata {statistics.median(five):.3f}s")
    print(f"median total {statistics.median(totals):.3f}s | CoT bocor {cot}/{len(prompts)}")
    print(f" fastest ttft {min(ttfts):.3f}s | MemAvail {mem()[0]:.0f} MB")


if __name__ == "__main__":
    main()