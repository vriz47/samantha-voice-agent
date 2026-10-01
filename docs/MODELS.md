# Model inventory

Weights are never committed (`models/` is gitignored). Fetch them with `fetch_model.sh` or
curl. Licences below were read from each archive's own `LICENSE` / `MODEL_CARD` / `README`.

## In use by the live loop

| Model | Role | Size on disk | Source | Licence |
|---|---|---|---|---|
| `sherpa-onnx-whisper-base.en` | STT (offline) | 433 MB | `./fetch_model.sh asr sherpa-onnx-whisper-base.en` | MIT |
| `silero_vad.onnx` | voice activity detection | 632 KB | `./fetch_model.sh asr silero_vad.onnx` | MIT |
| `sherpa-onnx-pocket-tts-int8-2026-01-26` | TTS, clones `bria.wav` | 195 MB | `./fetch_model.sh tts sherpa-onnx-pocket-tts-int8-2026-01-26` | CC-BY 4.0 text, README states **non-commercial only** |
| `smart-turn-v3` (pipecat-ai) | endpointing | 25 MB | `huggingface.co/pipecat-ai/smart-turn-v3` | BSD-2-Clause |

Smart Turn files used: `smart-turn-v3.0.onnx` (8,757,193 B),
`smart-turn-v3.2-cpu.onnx` (8,679,182 B). Official v3.2 benchmark: 92.63% accuracy over
31,527 samples across 23 languages; Indonesian 95.16%.

## Downloaded, waiting for the relay race

| Model | Why | Size | Source | Licence |
|---|---|---|---|---|
| `vits-piper-en_US-hfc_female-medium-int8` | candidate streaming TTS; sherpa VITS can emit `callback` chunks | 37 MB | `./fetch_model.sh tts vits-piper-en_US-hfc_female-medium-int8` | **CC-BY-NC-SA 4.0** (MODEL_CARD) |
| `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile` | streaming ASR with partial results, replaces 1.24 s Whisper pass | 110 MB | `./fetch_model.sh asr sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile` | Apache-2.0 |

Streaming callback behaviour for the Piper model is **not yet measured** — see
[docs/ARCHITECTURE.md](ARCHITECTURE.md#workaround-for-the-missing-streaming-tts).

## Evaluated and rejected as the primary voice

| Model | RTF (short / long) | Streams | Voices | Licence | Size |
|---|---|---|---|---|---|
| `kokoro-en-v0_19` | 2.57 / 2.38 | no | 11 fixed | Apache-2.0 | 354 MB |
| `sherpa-onnx-supertonic-3-tts-int8-2026-05-11` | 1.67 / 1.38 | no | 10 fixed | MIT (Supertone Inc.) | 139 MB |
| `kitten-nano-en-v0_8-int8` | 1.23 / 0.89 | no | 8 fixed, English only | Apache-2.0 | 45 MB |
| `kitten-micro-en-v0_8` | not run (int8 missing, fp32 only) | — | 8 fixed | Apache-2.0 | 62 MB |

Kitten nano is the fastest local engine and a reasonable fallback for English, but it cannot
clone a voice and would need a second engine for Indonesian.

## Voice references

`sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav` is the reference for
`VOICE_REFERENCE` in `samantha/engine.py`; `VOICE_STEPS = 8`. Cloning is only appropriate
for a recording whose owner consents to it.

## Non-model dependencies

| Dependency | Why |
|---|---|
| LiteRT-LM Gallery (`com.server.edge.gallery`) with `Gemma-4-E2B-it` | local LLM, OpenAI-compatible streaming endpoint |
| PulseAudio sink `OpenSL_ES_sink` | `paplay --raw` streaming playback |
| `libonnxruntime.so` from `site-packages/sherpa_onnx/lib` | runtime for Smart Turn (v1.15.1, API 15); Android has no `onnxruntime` Python wheel |