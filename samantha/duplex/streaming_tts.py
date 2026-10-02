"""Streaming VITS/Piper TTS engine.

Unlike Pocket-TTS (RTF 2.3-5.5), Piper generates faster than realtime (RTF 0.56-0.78), so
clause N+1 can be synthesised while clause N is still being spoken. Output is float32 mono
at the engine sample rate (22.05 kHz); :class:`StreamPlayer` resamples on the way out.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import sherpa_onnx as so

from samantha.duplex.profiler import Profiler

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
PIPER = os.path.join(MODELS, "vits-piper-en_US-hfc_female-medium-int8")
PIPER_MODEL = os.path.join(PIPER, "en_US-hfc_female-medium.onnx")
PIPER_LEXICON = os.path.join(PIPER, "en_US-hfc_female-medium.onnx.json")
PIPER_TOKENS = os.path.join(PIPER, "tokens.txt")
PIPER_DATA = os.path.join(PIPER, "espeak-ng-data")


class StreamingVoice:
    """Piper VITS with clause-at-a-time synthesis."""

    def __init__(self, threads: int = 6, speed: float = 0.80, prof: Profiler | None = None):
        if not os.path.isfile(PIPER_MODEL):
            raise FileNotFoundError(
                f"Piper TTS tidak ada di {PIPER}; jalankan: "
                "./fetch_model.sh tts vits-piper-en_US-hfc_female-medium-int8"
            )
        cfg = so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(
                    model=PIPER_MODEL,
                    lexicon=PIPER_LEXICON,
                    tokens=PIPER_TOKENS,
                    data_dir=PIPER_DATA,
                ),
                num_threads=threads,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        self.tts = so.OfflineTts(cfg)
        self.sample_rate = self.tts.sample_rate
        self.speed = speed
        self.prof = prof
        self._lock = threading.Lock()

    def speak(self, text: str) -> np.ndarray:
        """Synthesise one clause. Returns float32 mono."""
        out = self.tts.generate(text, sid=0, speed=self.speed)
        audio = np.asarray(out.samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError(f"Piper tidak menghasilkan audio untuk {text!r}")
        return audio

    def speak_streaming(self, text: str, player, first_only: bool = False) -> float:
        """Synthesise and push to a live player, measuring time to first pushed frame."""
        import time

        t0 = time.monotonic()
        audio = self.speak(text)
        gen_s = time.monotonic() - t0
        dur = len(audio) / self.sample_rate
        player.push(audio, self.sample_rate, text[-1:] if text else "")
        if self.prof is not None:
            self.prof.point(
                f"clause ready+queued: gen {gen_s:.3f}s -> {dur:.2f}s audio "
                f"(RTF {gen_s/dur:.2f})"
            )
        return gen_s