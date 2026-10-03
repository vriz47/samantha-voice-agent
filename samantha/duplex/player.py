"""Streaming PCM playback into a live PulseAudio sink.

``termux-media-player`` needs a finished file, which rules out the relay race. PulseAudio
accepts raw PCM on stdin, so one long-lived ``paplay --raw`` process per turn can receive
samples while they are still being produced.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time

import numpy as np

DEFAULT_DEVICE = "@DEFAULT_SINK@"
DEFAULT_RATE = 44100

XFADE_SEC = 0.030
PAUSE_COMMA_SEC = 0.12
PAUSE_STOP_SEC = 0.22
PEAK_TARGET = 0.70


class StreamPlayer:
    """One ``paplay --raw`` process per turn; push float32 mono, it is mixed to stereo."""

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        device: str = DEFAULT_DEVICE,
        latency_msec: int = 120,
        name: str = "samantha",
    ) -> None:
        self.rate = rate
        self.device = device
        self.latency_msec = latency_msec
        self.name = name
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._rc: dict[int, object] = {}
        self._queue: "queue.Queue[tuple[float, bytes] | None]" = queue.Queue(maxsize=64)
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self.first_audio_at: float | None = None
        self.frames_pushed = 0
        self.push_log: list[tuple[float, float]] = []
        self.drain_log: list[tuple[float, int, float]] = []
        self.first_drain_at: float | None = None
        self._tail = np.zeros(0, dtype=np.float32)

    def _resampler(self, src_rate: int):
        """Linear resampler; cached per source rate. TTS output is speech-band."""
        cached = self._rc.get(src_rate)
        if cached is None:

            def resample(x: np.ndarray) -> np.ndarray:
                if x.size == 0:
                    return x
                n_out = int(round(len(x) * self.rate / src_rate))
                if n_out <= 0:
                    return np.zeros(0, dtype=np.float32)
                pos = np.arange(n_out) * (src_rate / self.rate)
                i0 = np.floor(pos).astype(np.int64)
                frac = (pos - i0).astype(np.float32)
                i1 = np.minimum(i0 + 1, len(x) - 1)
                out = x[i0] * (1.0 - frac) + x[i1] * frac
                return out.astype(np.float32)

            self._rc[src_rate] = resample
            cached = resample
        return cached

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [
                "paplay",
                "--raw",
                f"--device={self.device}",
                "--format=s16le",
                f"--rate={self.rate}",
                "--channels=2",
                f"--latency-msec={self.latency_msec}",
                f"--client-name={self.name}",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._stop.clear()
        self._writer = threading.Thread(target=self._drain_loop, daemon=True)
        self._writer.start()

    def _drain_loop(self) -> None:
        """Write queued PCM to paplay without blocking the synthesiser."""
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return
            proc = self._proc
            if proc is None or proc.stdin is None:
                return
            queued_at, pcm = item
            handed_off = time.monotonic()
            if self.first_drain_at is None:
                self.first_drain_at = handed_off
            try:
                proc.stdin.write(pcm)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                return
            self.drain_log.append((handed_off, len(pcm) // 4, queued_at))

    def _prepare(self, samples: np.ndarray, src_rate: int, terminator: str) -> np.ndarray:
        """Resample, level-match and add the natural pause implied by the punctuation."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if src_rate != self.rate:
            audio = self._resampler(src_rate)(audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = audio * min(PEAK_TARGET / peak, 2.0)
        pause = 0.0
        if terminator:
            if terminator in ",;:":
                pause = PAUSE_COMMA_SEC
            elif terminator in ".!?":
                pause = PAUSE_STOP_SEC
        if pause:
            audio = np.concatenate([audio, np.zeros(int(pause * self.rate), dtype=np.float32)])
        return np.clip(audio, -1.0, 1.0)

    def _write(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        stereo = np.repeat(audio[:, None], 2, axis=1)
        pcm = (stereo * 32767.0).astype("<i2").tobytes()
        now_queued = time.monotonic()
        if self.first_audio_at is None:
            self.first_audio_at = now_queued
        self.frames_pushed += len(pcm) // 4
        self.push_log.append((now_queued, len(pcm) // 4))
        try:
            self._queue.put_nowait((now_queued, pcm))
        except queue.Full:
            return

    def push(self, samples: np.ndarray, src_rate: int, terminator: str = "") -> None:
        """Queue samples for playback. Never blocks the caller on playback pace.

        Fragments are crossfaded and given a punctuation-matched pause so separate TTS
        calls read as one continuous turn instead of a sequence of clipped utterances.
        """
        self.start()
        if samples is None or len(samples) == 0:
            return
        audio = self._prepare(samples, src_rate, terminator)
        n = min(len(self._tail), len(audio))
        if n:
            fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
            audio[:n] = self._tail[:n] * (1.0 - fade) + audio[:n] * fade
        keep = min(int(XFADE_SEC * self.rate), max(0, len(audio) - 1))
        if keep:
            self._tail = audio[-keep:].copy()
            audio = audio[:-keep]
        else:
            self._tail = np.zeros(0, dtype=np.float32)
        self._write(audio)

    def close(self, drain_timeout: float = 30.0) -> None:
        if self._tail.size:
            self._write(self._tail)
            self._tail = np.zeros(0, dtype=np.float32)
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._writer is not None:
            self._writer.join(timeout=2.0)
            self._writer = None
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        deadline = time.monotonic() + drain_timeout
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    def __enter__(self) -> "StreamPlayer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()