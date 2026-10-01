import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samantha.audio import AudioQueue
from samantha.duplex.profiler import Profiler
from samantha.engine import HOME, VOICE_STEPS, Samantha, log

PLAYERS = [
    ["termux-media-player", "play"],
    ["mpv", "--no-video", "--really-quiet"],
]


def find_player() -> list[str] | None:
    for cmd in PLAYERS:
        for p in os.environ.get("PATH", "").split(":"):
            if p and os.access(os.path.join(p, cmd[0]), os.X_OK):
                return cmd
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seconds", type=int, default=6)
    ap.add_argument("-n", "--turns", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--steps", type=int, default=VOICE_STEPS)
    ap.add_argument("--keep", action="store_true", help="jangan hapus file per giliran")
    ap.add_argument("--profile-json", help="tulis timeline profil ke file JSON")
    args = ap.parse_args()

    player = None if args.no_speak else find_player()
    if args.no_speak:
        log("mode diam: hanya tulis file")
    elif player is None:
        log("tidak ada player WAV")
    else:
        log(f"player: {' '.join(player)}")

    log("memuat Whisper + Pocket-TTS (sekali)")
    bot = Samantha(base_url=args.url)
    log("siap. ucup!")

    workdir = os.path.join(HOME, "turn")
    for i in range(args.turns):
        log(f"--- giliran {i + 1}/{args.turns} ---")
        if not args.keep and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)

        prof = Profiler(f"turn {i + 1}/{args.turns}")
        prof.point("start")
        info = bot.turn(seconds=args.seconds, synthesize=player is None, prof=prof)
        if info is None:
            continue

        if player is None:
            out = os.path.join(workdir, "all.wav")
            import soundfile as sf

            sf.write(out, info["audio"], info["sample_rate"], subtype="PCM_16")
            log(f"tulis {out}")
            continue

        log("meny synthesize per klausa (klausa 1 langsung main)")
        t_first = None
        queue = AudioQueue(player)
        t0 = time.time()
        total_audio = 0.0
        try:
            for clause, path, _s, _sr, secs in bot.voice.speak_clauses(
                info["reply"], workdir, steps=args.steps, prof=prof
            ):
                queue.put(path)
                prof.point("queue clause for playback")
                total_audio += secs
                if t_first is None:
                    t_first = time.time() - t0
                    prof.point(f"first audio starts ({t_first:.2f}s after text ready)")
                    log(f"klausa 1 siap {secs:.2f}s audio dalam {t_first:.2f}s -> main")
                else:
                    log(f"  klausa siap {secs:.2f}s (total {total_audio:.2f}s)")
            queue.drain()
        finally:
            queue.close()
        log(f"playback selesai: {total_audio:.2f}s audio, total {time.time()-t0:.2f}s")
        prof.point("turn done")
        log("\n" + prof.report())
        if args.profile_json:
            import json

            with open(args.profile_json, "w") as fh:
                json.dump(prof.to_dict(), fh, indent=2)
            log(f"profil -> {args.profile_json}")
    log("selesai")


if __name__ == "__main__":
    main()