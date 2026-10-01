"""Compare local TTS engines on load time, RTF, and whether they stream audio chunks.

Streaming (`callback fired > 1` with growing sample counts) is what makes Sesame's
relay race possible: text in, audio frames out before the sentence is finished.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SHORT = "Hi there, it's lovely to see you again."
LONG = (
    "I was thinking about the weekend and honestly I would love to spend "
    "some quiet time by the sea with you, if that sounds good to you."
)

CLONE_TEXT = "I think the best conversations happen when nobody is in a hurry."


def build_kokoro(name: str):
    import sherpa_onnx as so

    d = os.path.join(MODELS, name)
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            kokoro=so.OfflineTtsKokoroModelConfig(
                model=os.path.join(d, "model.onnx"),
                voices=os.path.join(d, "voices.bin"),
                tokens=os.path.join(d, "tokens.txt"),
                data_dir=os.path.join(d, "espeak-ng-data"),
            ),
            num_threads=6,
            provider="cpu",
        )
    )
    return so.OfflineTts(cfg)


def build_kitten(name: str):
    import sherpa_onnx as so

    d = os.path.join(MODELS, name)
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            kitten=so.OfflineTtsKittenModelConfig(
                model=os.path.join(d, "model.int8.onnx"),
                voices=os.path.join(d, "voices.bin"),
                tokens=os.path.join(d, "tokens.txt"),
                data_dir=os.path.join(d, "espeak-ng-data"),
            ),
            num_threads=6,
            provider="cpu",
        )
    )
    return so.OfflineTts(cfg)


def build_supertonic(name: str):
    import sherpa_onnx as so

    d = os.path.join(MODELS, name)
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            supertonic=so.OfflineTtsSupertonicModelConfig(
                duration_predictor=os.path.join(d, "duration_predictor.int8.onnx"),
                text_encoder=os.path.join(d, "text_encoder.int8.onnx"),
                vector_estimator=os.path.join(d, "vector_estimator.int8.onnx"),
                vocoder=os.path.join(d, "vocoder.int8.onnx"),
                tts_json=os.path.join(d, "tts.json"),
                unicode_indexer=os.path.join(d, "unicode_indexer.bin"),
                voice_style=os.path.join(d, "voice.bin"),
            ),
            num_threads=6,
            provider="cpu",
        )
    )
    return so.OfflineTts(cfg)


def build_clone(name: str, ref: str):
    import sherpa_onnx as so

    d = os.path.join(MODELS, name)
    samples, sr = sf.read(ref, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            pocket=so.OfflineTtsPocketModelConfig(
                model=os.path.join(d, "model.int8.onnx"),
                voices=os.path.join(d, "voices.bin"),
                tokens=os.path.join(d, "tokens.txt"),
                num_threads=6,
            )
        ),
        debug=False,
        provider="cpu",
    )
    tts = so.OfflineTts(cfg)
    gen = so.OfflineTtsPocketModelConfig() if False else None
    gcfg = so.GenerationConfig()
    gcfg.reference_audio = samples
    gcfg.reference_sample_rate = sr
    gcfg.reference_text = ""
    return tts, gcfg


def probe(tts, sid: int, label: str, streaming: bool = True) -> None:
    for tag, text in (("short", SHORT), ("long", LONG)):
        chunks: list[tuple[float, int, float]] = []
        t0 = time.monotonic()

        def cb(samples, progress):
            chunks.append((round(time.monotonic() - t0, 3), int(len(samples)), round(float(progress), 3)))
            return 0

        out = tts.generate(text, sid=sid, callback=cb if streaming else None)
        wall = time.monotonic() - t0
        dur = len(out.samples) / out.sample_rate
        first = chunks[0][0] if chunks else float("nan")
        print(
            f"  {label} sid={sid:<2} {tag:5s} wall {wall:6.2f}s audio {dur:5.2f}s "
            f"rtf {wall/dur:5.2f} chunks {len(chunks):3d} first_chunk {first:6.2f}s"
        )
        if chunks and len(chunks) > 1:
            print(f"      chunk timeline: {[(c[0], c[1]) for c in chunks[:8]]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="kokoro-en-v0_19,kitten-nano-en-v0_8-int8,kitten-micro-en-v0_8,"
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
    )
    ap.add_argument("--sids", default="0")
    ap.add_argument("--out", help="simpan contoh wav per model")
    args = ap.parse_args()

    sids = [int(s) for s in args.sids.split(",")]
    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===")
        t0 = time.monotonic()
        try:
            if name.startswith("kokoro"):
                tts = build_kokoro(name)
            elif name.startswith("kitten"):
                tts = build_kitten(name)
            elif name.startswith("sherpa-onnx-supertonic"):
                tts = build_supertonic(name)
            else:
                print("  builder tidak dikenal")
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  LOAD GAGAL: {type(exc).__name__}: {exc}")
            continue
        print(f"  load {time.monotonic()-t0:.2f}s  sr {tts.sample_rate}  speakers {tts.num_speakers}")
        for sid in sids:
            if sid >= tts.num_speakers:
                print(f"  sid {sid} di luar range")
                continue
            try:
                probe(tts, sid, name.split("-")[0])
            except Exception as exc:  # noqa: BLE001
                print(f"  sid {sid} GAGAL: {type(exc).__name__}: {exc}")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            out = tts.generate(SHORT, sid=sids[0])
            p = os.path.join(args.out, f"{name}-{sids[0]}.wav")
            sf.write(p, np.asarray(out.samples), out.sample_rate, subtype="PCM_16")
            print(f"  contoh -> {p}")


if __name__ == "__main__":
    main()