# Devlog

Findings that shaped the code, including the ones that failed.

## 2026-10-01 — profiling exposes the real bottleneck

Added `samantha/duplex/profiler.py` (monotonic stage spans + JSON timeline) and wired it
through `Samantha.turn()`, `Voice.speak_clauses()` and the CLI, plus
`tools/profile_turn.py` to replay a recorded turn without touching the mic.

The suspicion was the recording window. The measurement said otherwise: TTS generates at
**RTF 2.64**, taking 5.50 s to produce 2.08 s of speech, so end-of-speech → first audio was
**~9.2 s**, not the 2.4 s an earlier ad-hoc timing suggested. `num_steps` is not the lever —
4, 5, 6 and 8 steps all give RTF 2.3–2.4. Generation cost scales with text length.

`tools/tts_probe.py` then measured every local engine, and none of them stream: the
`callback(samples, progress)` API fires exactly once, after synthesis. That is the gap
against Sesame's relay race, and it is why clause-granularity relay is the v1 design.

## 2026-10-01 — relay race feasibility on Termux

Step 3 of the relay race (push audio frames into a playback queue) turned out to be
available: `paplay --raw --device=@DEFAULT_SINK@` accepted s16le PCM written 100 ms at a
time, rc=0. The `OpenSL_ES_sink` is live. Playback therefore does not need rewriting when a
streaming TTS arrives.

Step 2 was already in place: `BrainClient.stream()` yields SSE deltas. Measured first token
0.27 s warm, 1.26 s on a cold process; the 51-word system prompt and `max_tokens` are both
irrelevant to TTFT.

## 2026-10-01 — capture formats

* MP4/AAC cannot be read while recording: `moov atom not found`.
* FIFO is rejected before the recorder even starts: `File: pipe.mp4 already exists! Please
  specify a different filename`.
* **Raw AMR-WB grows incrementally** (16 kHz mono, 27.4 kbps) — the viable continuous
  capture path, with ~2 s MediaRecorder startup cost.

## 2026-09 — Smart Turn v3 integration

`pip install onnxruntime` has no Android wheel, so Smart Turn runs through a small C wrapper
(`native/onnx/ort_predict.c`) linked directly against sherpa's bundled
`libonnxruntime.so` — actual version 1.15.1, API 15, despite the symbol tag
`@@VERS_1.28.2`. Headers for API 15 are vendored.

Shape of the model: input `input_features [batch, 80, 800]` (8 s at 16 kHz — 800 frames, not
8000), output `logits [batch, 1]`, opset 18. `whisper_mel.py` is a port of the HF
`WhisperFeatureExtractor`.

Local results are ambiguous: v3.0 separates complete from mid-turn samples (0.936–0.959 vs
0.024–0.583) while v3.2 stays near-constant (0.945–0.987), yet v3.2's official benchmark is
stronger. Both return ~0.98 for silence, so VAD must gate the call. A labelled evaluation
set is still open work.

## 2026-09 — things that were fixed and kept

* Recording: `-l 0` plus our own `-q` timing; `_wait_decodable()` replaced a "stable file
  size" wait. Turn time went from 11.48 s to 8.50 s.
* STT: sherpa's `accept_waveform(sr, waveform)` argument order.
* TTS: the Pocket-TTS `GenerationConfig` call was wrong and caused runaway 39 s
  generations; the correct config plus 8 steps fixed it.
* Playback: `AudioQueue` starts clause N+1 while clause N plays, so no dead air between
  sentences.
* Persona: replies stopped leaking identity strings; 7 persona tests cover it.

## Dead ends

* **`sounddevice` / PortAudio / ALSA** — no audio backend exists in Termux on Android.
* **Native Android app as a prerequisite** — unnecessary for capture now that AMR-WB
  streams; keep it only as an `AudioRecord` fallback if 2 s MediaRecorder startup is too slow.
* **`max_tokens` tuning the LLM** — no effect on TTFT or total latency.
* **Shortening the system prompt** — under 40 ms of TTFT difference.