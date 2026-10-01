# Samantha

**A low-latency, fully on-device voice agent that runs entirely inside Termux on Android.**

No cloud services, no Android app, no GUI framework. A POCO F6 talks to itself: it listens,
transcribes on-device, reasons with a local LiteRT-LM model, and speaks back in a cloned
voice — while every stage is instrumented so the latency budget is a number, not a guess.

> **Status (2026-10-01):** the turn loop works end to end. Measured
> end-of-speech → first audio is **~9.2 s**, dominated by TTS (RTF 2.6). The relay-race
> rewrite that fixes this is specified in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
> its measurements live in [docs/LATENCY.md](docs/LATENCY.md).

---

## Why this exists

Voice assistants on Android are app-shaped: Kotlin, coroutines, `AudioRecord`, cloud TTS.
Samantha is the opposite constraint — a full duplex agent built from Python inside
Termux, where there is no ALSA, no PortAudio, and no raw-PCM capture from the Termux API.
Everything below was measured on the phone, not on a desktop.

## Pipeline

```
mic ──► VAD ──► STT ──► LLM (local) ──► clause split ──► TTS ──► playback
                                  │                                  │
                            persona guard                    mic stays live
                                                              for barge-in
```

| Stage | Component | Role |
|---|---|---|
| Capture | `termux-microphone-record` + `ffmpeg` | record, decode to 16 kHz mono PCM |
| VAD | Silero VAD (sherpa-onnx) | drop silence, split speech segments |
| STT | Whisper base.en int8 (sherpa-onnx) | transcript |
| Endpoint | Smart Turn v3 (ONNX) | complete vs mid-turn pause decision |
| LLM | LiteRT-LM Gallery, `Gemma-4-E2B-it` | reply text, streamed over SSE |
| Guard | `brain/persona.py` | identity-leak sanitiser, sentence clamp |
| TTS | Pocket-TTS int8 (voice clone `bria`) | speech |
| Playback | `AudioQueue` + `termux-media-player` | ordered, gapless clause playback |

The `Smart Turn` endpoint is wired as a library and benchmarked, but not yet in the live
turn loop — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#endpointing).

## Measured latency

From `tools/profile_turn.py` on a POCO F6 (Snapdragon 8s Gen 3, 6 threads), 5.63 s input
recording containing "Hello, how are you?":

| Stage | Time |
|---|---|
| model load (once per process) | 6.31 s |
| VAD | 0.13 s |
| STT | 1.24 s |
| LLM (buffered reply) | 2.50 s |
| TTS clause 1 | **5.50 s** → 2.08 s audio (RTF 2.64) |
| TTS clause 2 | 2.66 s → 0.96 s audio |
| **end-of-speech → first audio** | **~9.2 s** |

Goal: **<700 ms** to first audio, <9 s per turn. Full budget, per-engine RTF table and the
profiling method are in [docs/LATENCY.md](docs/LATENCY.md).

## Requirements

* Android with **Termux** and **Termux:API** (`pkg install termux-api`)
* `pkg install python ffmpeg clang`
* PulseAudio sink from Termux (`paplay`) for the streaming playback path
* Python deps: `pip install -r requirements.txt`
* A local LiteRT-LM Gallery server exposing an OpenAI-compatible `/v1/chat/completions`
  endpoint (`Gemma-4-E2B-it` used for all measurements)
* Models fetched with `./fetch_model.sh` — see [docs/MODELS.md](docs/MODELS.md)

## Quick start

```bash
cd ~/samantha

# 1. models (each is a single archive from the sherpa-onnx releases)
./fetch_model.sh tts sherpa-onnx-pocket-tts-int8-2026-01-26
./fetch_model.sh asr sherpa-onnx-whisper-base.en
./fetch_model.sh asr silero_vad.onnx

# Smart Turn v3 (BSD-2) comes from Hugging Face
mkdir -p models/smart-turn-v3 && cd models/smart-turn-v3
for f in smart-turn-v3.0.onnx smart-turn-v3.2-cpu.onnx; do
  curl -sLO "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/$f"
done
cd ~/samantha

# 2. talk to it (fixed recording window, then speak)
python samantha/cli.py --seconds 5

# 3. one turn only, no playback, for wiring checks
python samantha/cli.py --seconds 5 --no-speak
```

`--profile-json out.json` writes the full stage timeline for the turn.

## Tools

| Command | Purpose |
|---|---|
| `python samantha/cli.py -n 3` | three turns, clause-streamed playback |
| `python tools/profile_turn.py samantha_in.wav` | stage-by-stage latency of a recorded turn |
| `python tools/tts_probe.py` | RTF + streaming-callback check for every local TTS model |
| `python tools/chain_test.py` | end-to-end chain check |
| `bash native/onnx/build.sh` | build the C ONNX Runtime wrapper used by Smart Turn |

## Layout

```
brain/          LLM client (SSE, single-slot, cancel token), persona guard, chunker
samantha/       engine (capture → VAD → STT → LLM → TTS), CLI, audio queue
samantha/duplex/profiler.py    monotonic stage timing + JSON timeline
samantha/duplex/smart_turn.py  Smart Turn v3 via ctypes over sherpa's ONNX Runtime
samantha/duplex/whisper_mel.py Whisper-compatible log-mel frontend
native/onnx/    ort_predict.c: ONNX Runtime C API wrapper (Android has no Python wheel)
tools/          profiling and model probes
docs/           architecture, latency measurements, model inventory, devlog
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the relay race maps onto Termux, and why each component was chosen
* [docs/LATENCY.md](docs/LATENCY.md) — measurements, bottleneck analysis, path to sub-700 ms
* [docs/MODELS.md](docs/MODELS.md) — every model, its source, size and licence
* [docs/DEVLOG.md](docs/DEVLOG.md) — what was tried, what worked, what failed

## Licences

Source code: **MIT** — see [LICENSE](LICENSE).

Model weights are downloaded separately and keep their own licences; they are **not**
covered by the MIT licence. Two are **non-commercial**:

* Pocket-TTS — README states non-commercial use only
* `vits-piper-en_US-hfc_female` — CC-BY-NC-SA 4.0

Everything else is MIT (Whisper, Silero VAD), Apache-2.0 (Kokoro, Kitten TTS, streaming
zipformer) or BSD-2 (Smart Turn v3). Voice cloning is only appropriate for a recording whose
owner consents. Full table: [docs/MODELS.md](docs/MODELS.md).

## Acknowledgements

* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming/offline ASR, VAD, TTS, ONNX Runtime for Android
* [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) — open endpointing model (BSD-2)
* [LiteRT-LM Gallery](https://github.com/google-ai-edge/LiteRT-LM) — local LLM server
* Sesame's *Streaming Architecture for Low Latency* — the relay-race design this project is converging on