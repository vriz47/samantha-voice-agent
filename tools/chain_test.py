import os, subprocess, time, tempfile
import numpy as np
import soundfile as sf
import sherpa_onnx as so

HOME = "/data/data/com.termux/files/home"
D = f"{HOME}/samantha/models"
WD = f"{D}/sherpa-onnx-whisper-tiny.en"
POCK = f"{D}/sherpa-onnx-pocket-tts-int8-2026-01-26"


def record(path, seconds=8):
    for f in (path, path.replace(".m4a", ".wav")):
        if os.path.exists(f):
            os.remove(f)
    subprocess.run(
        ["termux-microphone-record", "-f", path, "-l", str(seconds)],
        capture_output=True, timeout=30,
    )
    time.sleep(seconds + 2.0)
    pcm = path.replace(".m4a", ".wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", pcm],
        capture_output=True, timeout=60,
    )
    return pcm


def vad_segments(samples, sr=16000, min_sec=0.4):
    c = so.VadModelConfig()
    c.silero_vad.model = f"{D}/silero_vad.onnx"
    c.sample_rate = sr
    vad = so.VoiceActivityDetector(c, buffer_size_in_seconds=30)
    segs = []
    step = 1600
    for i in range(0, len(samples), step):
        vad.accept_waveform(samples[i:i + step])
        while not vad.empty():
            segs.append(np.array(vad.front.samples, dtype=np.float32))
            vad.pop()
    return [s for s in segs if len(s) / sr >= min_sec]


def asr(model_dir, tok, threads=6):
    return so.OfflineRecognizer.from_whisper(
        encoder=f"{model_dir}/tiny.en-encoder.int8.onnx",
        decoder=f"{model_dir}/tiny.en-decoder.int8.onnx",
        tokens=tok, num_threads=threads, language="en", task="transcribe",
    )


def tts():
    p = so.OfflineTtsPocketModelConfig()
    p.encoder = f"{POCK}/encoder.onnx"
    p.decoder = f"{POCK}/decoder.int8.onnx"
    p.lm_main = f"{POCK}/lm_main.int8.onnx"
    p.lm_flow = f"{POCK}/lm_flow.int8.onnx"
    p.text_conditioner = f"{POCK}/text_conditioner.onnx"
    p.vocab_json = f"{POCK}/vocab.json"
    p.token_scores_json = f"{POCK}/token_scores.json"
    p.voice_embedding_cache_capacity = 8
    p.validate()
    m = so.OfflineTtsModelConfig()
    m.pocket = p
    m.num_threads = 6
    m.provider = "cpu"
    return so.OfflineTts(so.OfflineTtsConfig(model=m, max_num_sentences=1))


def main():
    rec = record(f"{HOME}/chain_in.m4a", 8)
    audio, sr = sf.read(rec, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"[1] rekaman : {len(audio)/sr:.2f}s @ {sr}Hz  rms={float(np.sqrt((audio**2).mean())):.4f}")

    segs = vad_segments(audio)
    print(f"[2] VAD     : {len(segs)} segmen >=0.4s -> "
          + ", ".join(f"{len(s)/sr:.2f}s" for s in segs))
    if not segs:
        print("    (tidak ada speech, stop)")
        return

    t = time.time()
    rec_ = asr(WD, f"{WD}/tiny.en-tokens.txt")
    print(f"[3] Whisper : load={time.time()-t:.1f}s")
    texts = []
    for s in segs:
        t = time.time()
        st = rec_.create_stream()
        st.accept_waveform(16000, s)
        rec_.decode_stream(st)
        el = time.time() - t
        txt = st.result.text.strip()
        texts.append(txt)
        print(f"    {len(s)/sr:.2f}s audio -> {el:.2f}s (RTF {el/(len(s)/sr):.2f}) | {txt[:110]}")
    heard = " ".join(texts).strip()
    print(f"[4] HEARD   : {heard!r}")

    if not heard:
        print("[5] TTS     : skip (tidak ada teks)")
        return

    t = time.time()
    tt = tts()
    ref, r_sr = sf.read(f"{POCK}/test_wavs/loona.wav", dtype="float32")
    if ref.ndim > 1:
        ref = ref[:, 0]
    g = so.GenerationConfig()
    g.reference_audio = ref
    g.reference_sample_rate = r_sr
    g.reference_text = ""
    g.num_steps = 4
    t = time.time()
    out = tt.generate(heard, g)
    el = time.time() - t
    d = len(out.samples) / out.sample_rate
    sf.write(f"{HOME}/chain_out.wav", out.samples, out.sample_rate, subtype="PCM_16")
    print(f"[5] TTS     : {d:.2f}s audio / {el:.2f}s = RTF {el/d:.2f} -> chain_out.wav")


main()