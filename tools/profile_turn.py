"""Latency breakdown of the full pipeline on a recorded wav, without touching the mic."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf

from samantha.audio import AudioQueue
from samantha.cli import find_player
from samantha.duplex.profiler import Profiler
from samantha.engine import HOME, VOICE_STEPS, Samantha, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default=os.path.join(HOME, "samantha_in.wav"))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--steps", type=int, default=VOICE_STEPS)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--json", help="tulis timeline ke file JSON")
    args = ap.parse_args()

    samples, sr = sf.read(args.wav, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    log(f"input {args.wav} = {len(samples)/sr:.2f}s")

    prof = Profiler("profile")
    bot = Samantha(base_url=args.url)

    prof.point("mic input ready (file)")
    with prof.span("vad"):
        segs = bot.voice.segments(samples)
    prof.point(f"segments: {len(segs)} ({', '.join(f'{len(s)/sr:.2f}s' for s in segs)})")
    if not segs:
        log("tidak ada speech di file ini")
        return

    with prof.span("stt"):
        heard = " ".join(bot.voice.transcribe(s) for s in segs).strip()
    prof.point(f"transcript ready: {heard!r}")

    with prof.span("llm"):
        reply = bot.brain.ask(
            bot.voice.turns
            + [
                {"role": "system", "content": __import__(
                    "brain.persona", fromlist=["SYSTEM"]).SYSTEM},
                {"role": "user", "content": heard},
            ],
            max_tokens=90,
        )
    prof.point("reply text ready")

    workdir = os.path.join(HOME, "turn")
    os.makedirs(workdir, exist_ok=True)

    if args.no_speak:
        with prof.span("tts_full"):
            audio, ar = bot.voice.speak(reply)
        prof.point(f"full TTS {len(audio)/ar:.2f}s audio")
    else:
        player = find_player()
        if player is None:
            log("tidak ada player; pakai --no-speak")
            return
        queue = AudioQueue(player)
        total = 0.0
        try:
            for clause, path, _s, _sr, secs in bot.voice.speak_clauses(
                reply, workdir, steps=args.steps, prof=prof
            ):
                queue.put(path)
                prof.point("clause queued -> playback running")
                total += secs
            queue.drain()
        finally:
            queue.close()
        prof.point(f"playback done ({total:.2f}s audio)")

    print()
    print(prof.report())
    first = next((t for t, n in prof.events if "first audio starts" in n), None)
    if first is not None:
        eos = next((t for t, n in prof.events if "transcript ready" in n), None)
        if eos is not None:
            print(f"\nend-of-speech -> first audio: {first - eos:.3f}s")
    if args.json:
        import json

        with open(args.json, "w") as fh:
            json.dump(prof.to_dict(), fh, indent=2)
        log(f"profil -> {args.json}")


if __name__ == "__main__":
    main()