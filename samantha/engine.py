import os
import re
import subprocess
import time

import numpy as np
import soundfile as sf
import sherpa_onnx as so

from brain.client import BrainClient, BrainTransportError

from brain.persona import SYSTEM, clamp_sentences, sanitize

from samantha.duplex.profiler import Profiler

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HOME, "models")
WHISPER = os.path.join(MODELS, "sherpa-onnx-whisper-base.en")
POCKET = os.path.join(MODELS, "sherpa-onnx-pocket-tts-int8-2026-01-26")
VAD_MODEL = os.path.join(MODELS, "silero_vad.onnx")

SR = 16000
SPEECH_SR = 24000
RECORD_BITRATE_KBPS = 128
VOICE_REFERENCE = "bria.wav"
VOICE_STEPS = 8


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"'“])")


def sentences_for_tts(text: str, max_len: int = 90) -> list[str]:
    """Split a reply at sentence boundaries so the first clause can play early.

    ``brain.chunker.clauses`` keeps short replies as one block on purpose (better
    TTS prosody), which would delay first audio, so streaming uses its own split.
    Over-long sentences are cut on commas before being forced back together.
    """
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        while len(part) > max_len:
            cut = max(part.rfind(", ", 0, max_len), part.rfind("; ", 0, max_len))
            if cut <= 0:
                cut = part.rfind(" ", 0, max_len)
            if cut <= 0:
                break
            out.append(part[:cut].strip())
            part = part[cut:].strip(" ,;")
        if part:
            out.append(part)
    return out or [text.strip()]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Voice:
    def __init__(self, threads: int = 6):
        tag = os.path.basename(WHISPER).replace("sherpa-onnx-whisper-", "")
        self._rec = so.OfflineRecognizer.from_whisper(
            encoder=os.path.join(WHISPER, f"{tag}-encoder.int8.onnx"),
            decoder=os.path.join(WHISPER, f"{tag}-decoder.int8.onnx"),
            tokens=os.path.join(WHISPER, f"{tag}-tokens.txt"),
            num_threads=threads,
            language="en",
            task="transcribe",
        )
        self._vad_cfg = so.VadModelConfig()
        self._vad_cfg.silero_vad.model = VAD_MODEL
        self._vad_cfg.sample_rate = SR
        self._tts = self._build_tts(threads)
        ref, r_sr = sf.read(
            os.path.join(POCKET, "test_wavs", VOICE_REFERENCE), dtype="float32"
        )
        if ref.ndim > 1:
            ref = ref[:, 0]
        self._ref = ref
        self._ref_sr = r_sr
        self._threads = threads
        self.turns: list[dict] = []

    @staticmethod
    def _build_tts(threads: int):
        p = so.OfflineTtsPocketModelConfig()
        p.encoder = os.path.join(POCKET, "encoder.onnx")
        p.decoder = os.path.join(POCKET, "decoder.int8.onnx")
        p.lm_main = os.path.join(POCKET, "lm_main.int8.onnx")
        p.lm_flow = os.path.join(POCKET, "lm_flow.int8.onnx")
        p.text_conditioner = os.path.join(POCKET, "text_conditioner.onnx")
        p.vocab_json = os.path.join(POCKET, "vocab.json")
        p.token_scores_json = os.path.join(POCKET, "token_scores.json")
        p.voice_embedding_cache_capacity = 8
        if not p.validate():
            raise RuntimeError("invalid Pocket-TTS config")
        m = so.OfflineTtsModelConfig()
        m.pocket = p
        m.num_threads = threads
        m.provider = "cpu"
        return so.OfflineTts(so.OfflineTtsConfig(model=m, max_num_sentences=1))

    # ---------- I/O ----------

    def record(self, seconds: int = 8, workdir: str = HOME) -> str:
        """Record ``seconds`` of audio and return a path to 16kHz mono PCM.

        ``termux-microphone-record`` only writes a playable file once the
        recorder stops (the MP4 ``moov`` atom lands at the end), so the live
        signal cannot be inspected while recording. Recording with ``-l 0`` and
        sending ``-q`` ourselves gives control over the window, and the file is
        decodable about 0.2s later.
        """
        src = os.path.join(workdir, "samantha_in.m4a")
        pcm = os.path.join(workdir, "samantha_in.wav")
        for f in (src, pcm):
            if os.path.exists(f):
                os.remove(f)
        subprocess.run(
            ["termux-microphone-record", "-f", src, "-l", "0",
             "-b", str(RECORD_BITRATE_KBPS), "-r", str(SR), "-c", "1"],
            capture_output=True,
            timeout=30,
        )
        time.sleep(seconds)
        try:
            subprocess.run(["termux-microphone-record", "-q"],
                           capture_output=True, timeout=20)
        except (subprocess.SubprocessError, OSError):
            pass
        self._wait_decodable(src, pcm)
        if not os.path.exists(pcm) or os.path.getsize(pcm) <= 44:
            raise RuntimeError("recording produced no PCM")
        return pcm

    @staticmethod
    def _wait_decodable(src: str, pcm: str, timeout: float = 8.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(pcm):
                os.remove(pcm)
            try:
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                     "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", pcm],
                    capture_output=True, timeout=60,
                )
            except (subprocess.SubprocessError, OSError):
                r = None
            size = os.path.getsize(pcm) if os.path.exists(pcm) else 0
            if r is not None and r.returncode == 0 and size > 44:
                return
            time.sleep(0.1)
        raise RuntimeError(f"recording never became decodable: {src}")

    @staticmethod
    def save_wav(samples: np.ndarray, sr: int, path: str) -> None:
        sf.write(path, samples, sr, subtype="PCM_16")

    # ---------- stages ----------

    def segments(self, samples: np.ndarray, min_sec: float = 0.4) -> list[np.ndarray]:
        vad = so.VoiceActivityDetector(self._vad_cfg, buffer_size_in_seconds=30)
        out: list[np.ndarray] = []
        step = 1600
        for i in range(0, len(samples), step):
            vad.accept_waveform(samples[i:i + step])
            while not vad.empty():
                out.append(np.array(vad.front.samples, dtype=np.float32))
                vad.pop()
        return [s for s in out if len(s) / SR >= min_sec]

    def transcribe(self, samples: np.ndarray) -> str:
        st = self._rec.create_stream()
        st.accept_waveform(SR, samples)
        self._rec.decode_stream(st)
        return st.result.text.strip()

    def speak(self, text: str, speed: float = 1.0, steps: int = VOICE_STEPS):
        cfg = so.GenerationConfig()
        cfg.reference_audio = self._ref
        cfg.reference_sample_rate = self._ref_sr
        cfg.reference_text = ""
        cfg.num_steps = steps
        cfg.speed = speed
        out = self._tts.generate(text, cfg)
        if len(out.samples) == 0:
            raise RuntimeError("TTS produced no audio")
        return out.samples, out.sample_rate

    def speak_clauses(self, text: str, outdir: str, steps: int = VOICE_STEPS, prof=None):
        """Synthesise clause by clause so the first clause plays early.

        Yields ``(clause, wav_path, samples, sample_rate, seconds)`` as each clause
        finishes. Audio for clause N+1 is generated while clause N is playing.
        """
        parts = sentences_for_tts(text)
        for i, part in enumerate(parts):
            t = time.time()
            samples, sr = self.speak(part, steps=steps)
            gen_s = time.time() - t
            path = os.path.join(outdir, f"clause_{i:02d}.wav")
            sf.write(path, samples, sr, subtype="PCM_16")
            if prof is not None:
                prof.point(f"clause {i} ready ({gen_s:.2f}s gen, {len(samples)/sr:.2f}s audio)")
            yield part, path, samples, sr, len(samples) / sr

    def rtf(self, text: str, steps: int = VOICE_STEPS) -> tuple[float, float]:
        t = time.time()
        s, sr = self.speak(text, steps=steps)
        el = time.time() - t
        dur = len(s) / sr
        return dur, el / dur


class Samantha:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", threads: int = 6):
        self.voice = Voice(threads=threads)
        self.brain = BrainClient(base_url=base_url)

    def turn(self, seconds: int = 8, synthesize: bool = True, prof=None) -> dict | None:
        if prof is None:
            prof = Profiler("turn")
        prof.point("mic open")
        with prof.span("record"):
            pcm = self.voice.record(seconds=seconds)
        prof.point("mic closed")
        with prof.span("decode"):
            samples, sr = sf.read(pcm, dtype="float32")
            if samples.ndim > 1:
                samples = samples[:, 0]

        with prof.span("vad"):
            segs = self.voice.segments(samples)
        if not segs:
            log("tidak ada speech")
            return None
        prof.point(f"speech segments: {len(segs)}")

        with prof.span("stt"):
            heard = " ".join(self.voice.transcribe(s) for s in segs).strip()
        stt_s = prof.durations.get("stt", 0.0)
        log(f"HEARD ({stt_s:.2f}s): {heard!r}")
        if not heard:
            return None

        with prof.span("llm"):
            try:
                reply = self.brain.ask(
                    self.voice.turns + [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": heard},
                    ],
                    max_tokens=90,
                )
            except (BrainTransportError, TimeoutError) as exc:
                log(f"LLM gagal: {type(exc).__name__}: {exc}")
                reply = "Sorry, I lost connection for a second. Say that again?"
        llm_s = prof.durations.get("llm", 0.0)
        raw = reply
        with prof.span("postprocess"):
            reply = clamp_sentences(sanitize(reply))
        if not reply:
            reply = "Hmm, let me think about that. Tell me a bit more."
        if reply != raw:
            log(f"REPLY ({llm_s:.2f}s): {reply!r}  [was {raw!r}]")
        else:
            log(f"REPLY ({llm_s:.2f}s): {reply!r}")
        prof.point("reply text ready")

        audio, ar = None, None
        tts_s = 0.0
        dur = 0.0
        if synthesize:
            with prof.span("tts"):
                audio, ar = self.voice.speak(reply)
            tts_s = prof.durations.get("tts", 0.0)
            dur = len(audio) / ar

        self.voice.turns.extend(
            [{"role": "user", "content": heard}, {"role": "assistant", "content": reply}]
        )

        info = {
            "heard": heard,
            "reply": reply,
            "record_s": prof.durations.get("record", 0.0),
            "decode_s": prof.durations.get("decode", 0.0),
            "vad_s": prof.durations.get("vad", 0.0),
            "stt_s": stt_s,
            "llm_s": llm_s,
            "tts_s": tts_s,
            "audio_s": dur,
            "tts_rtf": (tts_s / dur) if dur else 0.0,
            "total_s": prof.elapsed,
            "audio": audio,
            "sample_rate": ar,
            "prof": prof,
        }
        if synthesize:
            log(
                f"TTS {dur:.2f}s audio / {tts_s:.2f}s (RTF {tts_s/dur:.2f}) | "
                f"total {info['total_s']:.2f}s"
            )
        else:
            log(f"total {info['total_s']:.2f}s")
        return info