# Architecture

Two layers: the loop that runs today (`v0`) and the relay-race loop being built (`v1`).

## v0 — the current turn loop

`samantha/engine.py::Samantha.turn()`

1. **Capture** — `Voice.record()` starts `termux-microphone-record -l 0`, sleeps for a
   fixed window, sends `-q`, then polls `ffmpeg` until the file decodes.
2. **VAD** — `Voice.segments()` feeds 100 ms chunks to sherpa's Silero VAD and returns
   speech regions of at least 0.4 s.
3. **STT** — `Voice.transcribe()` runs Whisper base.en per segment.
4. **LLM** — `BrainClient.ask()` drains the SSE stream to `[DONE]` and returns the whole
   reply, so no text is visible to TTS until generation ends.
5. **Guard** — `sanitize()` then `clamp_sentences()`.
6. **TTS + playback** — `speak_clauses()` synthesises sentence by sentence;
   `AudioQueue` hands each finished WAV to a player in order, so clause N+1 is generated
   while clause N plays.

Stage timings come from `samantha/duplex/profiler.py`.

### Why a fixed recording window

`termux-microphone-record` writes MP4/AAC, and the `moov` atom only lands when recording
stops, so `ffmpeg` cannot read the file while it is still growing:

```
ffprobe: moov atom not found
```

Two alternatives were tested and rejected for v0:

* **FIFO / named pipe** — Termux:API refuses an existing path before it even opens the
  recorder: `File: pipe.mp4 already exists! Please specify a different filename`.
* **Raw AMR-WB** (`-e amr_wb -b 128 -r 16000 -c 1`) — *does* grow incrementally
  (12.7 kB after 5 s, decodes as `amr_wb`, 16 kHz mono, 27.4 kbps). This is the escape
  hatch for continuous capture; a native `AudioRecord` shim is the other.

## v1 — relay race

Target design follows Sesame's *Streaming Architecture for Low Latency*: no stage waits for
the previous one to finish.

```
mic thread ──► VAD ──► endpoint ──► STT ──► LLM stream ──► clause queue ──► TTS ──► paplay
   │                (Smart Turn)         (SSE)              (sentence)        (raw PCM)
   └──────────────── barge-in monitor, always live ───────────────────────────┘
```

Status of each step, verified on the phone:

| Step | Status | Evidence |
|---|---|---|
| 2. Feed LLM text chunks into TTS as they arrive | **ready** | `BrainClient.stream()` already yields deltas; measured first token 0.27 s warm |
| 3. Push audio frames into a playback queue | **ready** | `paplay --raw --device=@DEFAULT_SINK@` accepted PCM written in 100 ms chunks (rc=0); sink `OpenSL_ES_sink` is live |
| 4. Concurrent stages + shared cancellation | **partial** | `CancelToken` exists in `brain/client.py`; must be plumbed into capture, TTS and playback |
| 1. One persistent streaming TTS connection | **missing** | every local TTS backend fires its `callback` exactly once, after synthesis completes |

### Workaround for the missing streaming TTS

`OfflineTts.generate()` accepts `callback(samples, progress)`, which would allow pushing
frames as they are produced. Measured behaviour of all local engines:

| Engine | callback invocations | short clause RTF | long text RTF |
|---|---|---|---|
| Pocket-TTS (cloned `bria`) | 1, after 6.63 s | 2.34 | 5.53 |
| Kokoro v0.19 | 1, at end | 2.57 | 2.38 |
| Supertonic 3 | 1, at end | 1.67 | 1.38 |
| Kitten nano int8 | 1, at end | 1.23 | 0.89 |

So v1 gates on **clause granularity** rather than token granularity: one `paplay --raw`
process lives for the whole turn, clause 1 is pushed as soon as it is synthesised, and
clause 2 is synthesised while clause 1 is audible. Because playback already consumes raw
PCM, switching to a genuinely streaming engine (VITS/Matcha emit per-chunk callbacks in
sherpa-onnx) is a drop-in change with no architecture edit.

Downloaded and ready for that test: `vits-piper-en_US-hfc_female-medium-int8` (20 MB,
CC-BY-NC-SA 4.0 — non-commercial) and `sherpa-onnx-streaming-zipformer-en-20M` (Apache-2.0).

## Endpointing

`samantha/duplex/smart_turn.py` loads Smart Turn v3 through a ctypes wrapper over sherpa's
bundled `libonnxruntime.so`, because Android has no `onnxruntime` Python wheel. Input is a
Whisper-compatible log-mel window (`whisper_mel.py`), 8 s @ 16 kHz = `[80, 800]`.

Local behaviour so far:

| Model | complete sample | mid-turn sample | silence | inference (4 threads) |
|---|---|---|---|---|
| v3.0 | 0.936–0.959 | 0.024–0.583 | 0.976 | 98 ms |
| v3.2-cpu | 0.945–0.987 | 0.939–0.984 | 0.987 | 112 ms |

v3.0 discriminates better on the available samples; v3.2's official benchmark is stronger
(92.63% overall accuracy, 15 languages benchmarked per its model card). Both saturate on
silence, which is why VAD must gate the call rather than treating `p >= 0.5` as sufficient.
A labelled evaluation set is still needed before choosing.

## Cancellation contract

The LiteRT-LM Gallery backend (Ktor/Netty) crashes with `ClosedWriteChannelException`
when a client closes the socket mid-stream. A cancelled turn is therefore always drained to
`[DONE]` server-side and only the client-side output is discarded. `BrainClient.stream()`
implements this with a worker thread that owns the socket; v1 must reuse the same pattern
for TTS and playback so an interrupted turn can drop queued audio without killing the
server.

## Barge-in

Not implemented. The pieces needed:

* capture thread stays live during playback (raw AMR-WB or native `AudioRecord`)
* echo cancellation, otherwise the agent hears its own voice and interrupts itself
* backchannel filter ("mm-hm", "uh-huh") so it does not treat politeness as a command

## Threading model today

| Thread | Work | Sync |
|---|---|---|
| main | `turn()` stages in order | — |
| brain worker | owns the HTTP socket, pushes deltas to a `Queue` | `_slot` lock (server is single-slot) |
| audio queue | `AudioQueue` plays one WAV at a time | `Queue` of file paths |

v1 adds: capture thread, TTS thread per clause, one long-lived `paplay` writer, all reading
from the same `CancelToken`.