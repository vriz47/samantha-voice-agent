"""Whisper log-mel front-end, ported to match HuggingFace WhisperFeatureExtractor.

smart-turn-v3 consumes the same 80-bin log-mel spectrogram as Whisper's encoder
(80 bins, 400-pt FFT, 160-sample hop, 16 kHz, 8 s window = 800 frames), so the
front-end must be reproduced exactly rather than approximated.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
WINDOW_SECONDS = 8.0
N_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
N_FRAMES = N_SAMPLES // HOP_LENGTH
MEL_FLOOR = 1e-10

_MEL_MIN_HERTZ = 0.0
_MEL_MAX_HERTZ = 8000.0

_FILTER_BANK: np.ndarray | None = None
_WINDOW: np.ndarray | None = None


def _hz_to_mel_slaney(freq):
    freq = np.asarray(freq, dtype=np.float64)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    if np.any(log_region):
        safe = np.where(log_region, freq, 1.0)
        mels = np.where(log_region, min_log_mel + np.log(safe / min_log_hertz) * logstep, mels)
    return mels


def _mel_to_hertz_slaney(mels):
    mels = np.asarray(mels, dtype=np.float64)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    if np.any(log_region):
        freq = np.where(log_region, min_log_hertz * np.exp(logstep * (mels - min_log_mel)), freq)
    return freq


def _build_filter_bank() -> np.ndarray:
    num_bins = 1 + N_FFT // 2
    mel_freqs = np.linspace(
        _hz_to_mel_slaney(_MEL_MIN_HERTZ), _hz_to_mel_slaney(_MEL_MAX_HERTZ), N_MELS + 2
    )
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    bank = np.maximum(0.0, np.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (filter_freqs[2 : N_MELS + 2] - filter_freqs[:N_MELS])
    return bank * enorm[None, :]


def _filter_bank() -> np.ndarray:
    global _FILTER_BANK
    if _FILTER_BANK is None:
        _FILTER_BANK = _build_filter_bank()
    return _FILTER_BANK


def _hann() -> np.ndarray:
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = np.hanning(N_FFT + 1)[:-1]
    return _WINDOW


def fit_window(audio: np.ndarray) -> np.ndarray:
    """Trim to the last 8 s or left-pad with zeros, matching smart-turn's helper."""
    a = np.asarray(audio, dtype=np.float64).reshape(-1)
    if a.size > N_SAMPLES:
        return a[-N_SAMPLES:]
    if a.size < N_SAMPLES:
        return np.pad(a, (N_SAMPLES - a.size, 0))
    return a


def log_mel(audio_16k: np.ndarray) -> np.ndarray:
    """16 kHz mono float audio -> (80, 8000) float32 log-mel, Whisper-normalised."""
    x = fit_window(audio_16k)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)

    padded = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    num_frames = 1 + (padded.size - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH][:num_frames]

    spec = np.fft.rfft(frames * _hann(), axis=-1)
    power = (np.abs(spec) ** 2).T

    mel = np.maximum(MEL_FLOOR, _filter_bank().T @ power)
    log_spec = np.log10(mel)
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)