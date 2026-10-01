# Latency

Every number here was measured on the target device: **POCO F6, Snapdragon 8s Gen 3, 6
sherpa-onnx threads, Android 16, Termux**. Targets: **<700 ms** from end-of-speech to first
audio, **<9 s** per turn.

## How to measure

```bash
python tools/profile_turn.py samantha_in.wav          # stage breakdown of a recorded turn
python tools/profile_turn.py samantha_in.wav --json out.json
python samantha/cli.py -n 1 --profile-json turn.json  # live mic turn
python tools/tts_probe.py                             # per-engine RTF + streaming check
```

`samantha/duplex/profiler.py` records monotonic timestamps per stage plus a timeline of
events, and `Samantha.turn()` returns per-stage durations in its result dict
(`record_s`, `decode_s`, `vad_s`, `stt_s`, `llm_s`, `tts_s`, `tts_rtf`, `total_s`).

## Turn profile

Input `samantha_in.wav`, 5.63 s, transcript *"Hello, how are you?"*.

| Stage | Time |
|---|---|
| model load (once per process) | 6.31 s |
| VAD | 0.13 s |
| STT | 1.24 s |
| LLM (reply buffered to the end) | 2.50 s |
| TTS clause 1 | 5.50 s → 2.08 s audio |
| TTS clause 2 | 2.66 s → 0.96 s audio |
| **end-of-speech → first audio** | **~9.2 s** |

**TTS is the bottleneck, not the microphone.** It generates slower than realtime (RTF 2.64
on the first clause), which alone makes a sub-second first audio impossible.

## TTS engines

`tools/tts_probe.py`, 6 threads:

| Engine | load | short clause | long text | streams? | notes |
|---|---|---|---|---|---|
| Pocket-TTS int8 | 1.9 s | RTF 2.34 | RTF 5.53 | no | clones `bria`; multilingual; non-commercial |
| Kokoro v0.19 | 4.9 s | RTF 2.57 | RTF 2.38 | no | 11 fixed speakers |
| Supertonic 3 int8 | 2.6 s | RTF 1.67 | RTF 1.38 | no | 10 fixed speakers, 44.1 kHz |
| Kitten nano int8 | 2.8 s | **RTF 1.23** | **RTF 0.89** | no | 8 fixed speakers, English |

`num_steps` is not the lever: Pocket-TTS at 4, 5, 6 and 8 steps all land at RTF 2.3–2.4.
Generation cost scales with text length, so the first clause should be short.

`callback(samples, progress)` fires **once**, after synthesis, for every engine — e.g.
Pocket-TTS on a 103-character sentence: 6.63 s wall, one callback carrying 28 800 samples.
There is no partial audio to forward.

## LLM token timeline

`Gemma-4-E2B-it` on LiteRT-LM Gallery, `max_tokens` 12 vs 90 makes no difference:

| Milestone | Cold process | Warm |
|---|---|---|
| first token | 1.26 s | 0.27–0.30 s |
| 5 words | 1.60 s | — |
| 10 words | 2.01 s | — |
| first sentence complete | 2.63 s | — |
| two-sentence reply finished | 3.54 s | 2.03–2.61 s |

Cold start costs ~1 s. Prefill is not the problem — the 51-word system prompt changes TTFT by
under 40 ms. `max_tokens` does not change TTFT either.

## Capture

| Format | Behaviour |
|---|---|
| MP4/AAC (`-e aac`, default) | `moov atom not found` until the recorder stops |
| FIFO | rejected up front: `File: pipe.mp4 already exists!` |
| AMR-WB (`-e amr_wb -b 128 -r 16000 -c 1`) | grows incrementally; 1.26 kB at 3 s, 12.7 kB at 5 s; 16 kHz mono, 27.4 kbps |
| AMR-NB (`-e amr_nb`) | grows after ~1 s; 8 kHz mono |

AMR-WB MediaRecorder startup costs about **2 s** before the first bytes appear — too slow for
a 700 ms budget on its own, which is why a native `AudioRecord` shim stays on the table.

## Playback

`termux-media-player play file.wav` works (2.21 s audio in ~2 s wall) but needs a complete
file, so it cannot express streaming.

PulseAudio is available: sink `OpenSL_ES_sink`, s16le 2ch 44.1 kHz.

```bash
paplay --raw --device=@DEFAULT_SINK@ --format=s16le --rate=44100 --channels=2 < chunks
```

Feeding 100 ms chunks 150 ms apart (simulated RTF 0.67) exits rc=0 with no errors — the
write-and-playback loop works and is the sink v1 will use.

## Path to <700 ms

| Stage | now | target | lever |
|---|---|---|---|
| capture | fixed 5 s window | ~0 | AMR-WB tail or native `AudioRecord` |
| STT | 1.24 s | 0.1–0.3 s | streaming zipformer with partial transcripts; Whisper kept as final pass |
| LLM → first speakable chunk | 1.60 s (cold) | 0.4–0.6 s | stream deltas, cut on first clause; prewarm the server |
| TTS → first frame | 5.50 s | 0.15–0.30 s | streaming-capable engine (VITS/Piper; downloaded, untested) |
| playback buffer | file-at-once | 120 ms | `paplay --raw` with `--latency-msec=120` |

Sum of the targets is roughly 0.9 s; reaching 700 ms needs the streaming TTS *and* streaming
ASR, in which case STT and LLM overlap and the effective critical path shortens further.