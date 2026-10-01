"""Smart Turn v3 semantic end-of-turn detector, run on-device via sherpa's onnxruntime."""

from __future__ import annotations

import ctypes
import os

import numpy as np

from .whisper_mel import N_FRAMES, N_MELS, SAMPLE_RATE, log_mel

_DEFAULT_LIB = os.path.expanduser("~/samantha/native/onnx/build/libsamantha_ort.so")
_DEFAULT_MODEL = os.path.expanduser("~/samantha/models/smart-turn-v3/smart-turn-v3.2-cpu.onnx")

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class SmartTurnUnavailable(RuntimeError):
    pass


class SmartTurn:
    """Wraps libsamantha_ort.so: probability that the speaker finished their turn.

    The model is a Whisper-tiny encoder plus a linear head. It sees raw prosody
    (no transcript), so it can commit a turn later than a text-based endpoint when
    the speaker pauses mid-thought.
    """

    def __init__(self, model: str | None = None, lib: str | None = None, threads: int = 1) -> None:
        model_path = model or _DEFAULT_MODEL
        lib_path = lib or os.environ.get("SAMANTHA_ORT_LIB") or _DEFAULT_LIB
        if not os.path.exists(model_path):
            raise SmartTurnUnavailable(f"smart-turn model not found: {model_path}")
        if not os.path.exists(lib_path):
            raise SmartTurnUnavailable(
                f"native ort wrapper not built: {lib_path} (run native/onnx/build.sh)"
            )

        self._lib = ctypes.CDLL(lib_path)
        self._lib.samantha_ort_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.samantha_ort_open.restype = ctypes.c_int
        self._lib.samantha_ort_predict.argtypes = [_FLOAT_PTR]
        self._lib.samantha_ort_predict.restype = ctypes.c_float
        self._lib.samantha_ort_ready.restype = ctypes.c_int
        self._lib.samantha_ort_close.restype = None

        rc = self._lib.samantha_ort_open(model_path.encode(), int(threads))
        if rc != 0:
            raise SmartTurnUnavailable(f"samantha_ort_open failed (rc={rc}) for {model_path}")

    def probability(self, audio_16k: np.ndarray) -> float:
        """P(turn complete) for up to the last 8 s of 16 kHz mono audio."""
        feats = np.ascontiguousarray(log_mel(audio_16k), dtype=np.float32)
        if feats.shape != (N_MELS, N_FRAMES):
            raise ValueError(f"unexpected feature shape {feats.shape}")
        return float(self._lib.samantha_ort_predict(feats.ctypes.data_as(_FLOAT_PTR)))

    def is_complete(self, audio_16k: np.ndarray, threshold: float = 0.5) -> bool:
        return self.probability(audio_16k) >= threshold

    def scan(self, audio_16k: np.ndarray, hop_seconds: float = 0.2, threshold: float = 0.5):
        """Scan forward over the audio, yielding (end_seconds, probability, complete).

        Each window is the most recent 8 s ending at `end_seconds`, matching how the
        model was trained to see an utterance tail.
        """
        a = np.asarray(audio_16k, dtype=np.float64).reshape(-1)
        step = max(1, int(SAMPLE_RATE * hop_seconds))
        for end in range(step, a.size + 1, step):
            window = a[max(0, end - N_FRAMES * 160) : end]
            p = self.probability(window)
            yield end / SAMPLE_RATE, p, p >= threshold

    def close(self) -> None:
        self._lib.samantha_ort_close()