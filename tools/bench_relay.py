"""Clean relay-race benchmark.

Run detached so the caller can go idle and stop competing for the 8 CPU cores:

    nohup python tools/bench_relay.py > /tmp/bench_relay.out 2>&1 &
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.client import BrainClient  # noqa: E402
from brain.persona import SYSTEM  # noqa: E402
from samantha.duplex.profiler import Profiler  # noqa: E402
from samantha.duplex.relay import RelayTurn  # noqa: E402
from samantha.engine import Voice, log  # noqa: E402

QUESTIONS = [
    "Tell me about the sea.",
    "What is your favourite season?",
    "Say something about the moon.",
    "How do you feel when it rains?",
    "Explain what a rainbow is.",
]


def main() -> None:
    tts_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=10) as r:
        model = json.load(r)["data"][0]["id"]

    log(f"model: {model}")
    log(f"muat Whisper + Piper dengan {tts_threads} thread TTS")
    voice = Voice(threads=tts_threads)
    brain = BrainClient(model=model)
    prof = Profiler(f"bench-relay-t{tts_threads}")
    relay = RelayTurn(voice, brain, tts_threads=tts_threads, prof=prof)

    rows = []
    for round_no in range(rounds):
        for q in QUESTIONS:
            t0 = time.monotonic()
            info = relay.run(q, seconds=5)
            wall = time.monotonic() - t0
            first = info["first_audio_s"]
            rows.append(
                {
                    "round": round_no,
                    "q": q,
                    "first_audio_s": first,
                    "wall_s": wall,
                    "clauses": len(info["clauses"]),
                    "errors": info["errors"],
                }
            )
            shown = f"{first:.3f}s" if first is not None else "TIDAK ADA"
            log(
                f"r{round_no} {q[:34]!r} first={shown} "
                f"wall={wall:.2f}s clauses={len(info['clauses'])} err={info['errors'] or 'none'}"
            )

    first = [r["first_audio_s"] for r in rows if r["first_audio_s"]]
    walls = [r["wall_s"] for r in rows]
    errors = [r for r in rows if r["errors"]]
    summary = {
        "tts_threads": tts_threads,
        "rounds": rounds,
        "n": len(rows),
        "first_audio_min": min(first),
        "first_audio_median": statistics.median(first),
        "first_audio_mean": statistics.mean(first),
        "first_audio_max": max(first),
        "first_audio_p90": sorted(first)[int(len(first) * 0.9) - 1],
        "wall_median": statistics.median(walls),
        "under_700ms": sum(1 for f in first if f < 0.7),
        "under_1s": sum(1 for f in first if f < 1.0),
        "errors": len(errors),
        "rows": rows,
    }
    log("\n=== RINGKASAN ===")
    log(
        f"first audio: min {summary['first_audio_min']:.3f}s | median "
        f"{summary['first_audio_median']:.3f}s | mean {summary['first_audio_mean']:.3f}s | "
        f"max {summary['first_audio_max']:.3f}s"
    )
    log(
        f"<700ms: {summary['under_700ms']}/{len(rows)} | <1s: {summary['under_1s']}/{len(rows)} | "
        f"wall median {summary['wall_median']:.2f}s | errors {summary['errors']}"
    )
    Path("docs").mkdir(exist_ok=True)
    out = Path("docs") / f"bench-relay-t{tts_threads}.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"tulis {out}")
    print(prof.report())


if __name__ == "__main__":
    main()