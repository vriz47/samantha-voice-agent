"""Kokoro TTS engine for Samantha.

One process-wide engine per model: loading model.onnx costs seconds, so the
voice list is loaded once and reused for every clause.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import sherpa_onnx

DEFAULT_MODEL_DIR = Path("models/kokoro-en-v0_19")

_lock = threading.Lock()
_cache: dict[tuple, sherpa_onnx.OfflineTts] = {}


def build_config(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    voice: str = "af_heart",
    lang: str = "en-us",
    length_scale: float = 1.0,
    num_threads: int = 4,
) -> sherpa_onnx.OfflineTtsConfig:
    base = Path(model_dir)
    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig()
    kokoro.model = str(base / "model.onnx")
    kokoro.voices = str(base / "voices.bin")
    kokoro.tokens = str(base / "tokens.txt")
    kokoro.data_dir = str(base / "espeak-ng-data")
    kokoro.lang = lang
    kokoro.length_scale = length_scale
    kokoro.validate()

    model = sherpa_onnx.OfflineTtsModelConfig()
    model.kokoro = kokoro
    model.num_threads = num_threads
    model.provider = "cpu"

    cfg = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
    cfg.validate()
    return cfg


def get_tts(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    voice: str = "af_heart",
    lang: str = "en-us",
    length_scale: float = 1.0,
    num_threads: int = 4,
) -> tuple[sherpa_onnx.OfflineTts, float]:
    key = (str(model_dir), voice, lang, length_scale, num_threads)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit, 0.0
        start = time.time()
        cfg = build_config(model_dir, voice, lang, length_scale, num_threads)
        tts = sherpa_onnx.OfflineTts(cfg)
        if tts.sample_rate <= 0:
            raise RuntimeError("TTS engine reported an invalid sample rate")
        _cache[key] = tts
        return tts, time.time() - start


def voices(model_dir: Path | str = DEFAULT_MODEL_DIR) -> list[str]:
    cfg = build_config(model_dir, voice="af_heart")
    tts = sherpa_onnx.OfflineTts(cfg)
    return list(tts.voices)


def synthesize(
    tts: sherpa_onnx.OfflineTts,
    text: str,
    sid: int = 0,
    speed: float = 1.0,
) -> tuple[list[float], int]:
    audio = tts.generate(text, sid=sid, speed=speed)
    return audio.samples, audio.sample_rate


def seconds(samples: list[float], sample_rate: int) -> float:
    return len(samples) / float(sample_rate) if sample_rate else 0.0