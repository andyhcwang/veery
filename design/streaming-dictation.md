# Streaming Dictation Design (researched 2026-07-18)

Research basis: ufal/whisper_streaming (LocalAgreement-2), Argmax WhisperKit eager mode,
whisper.cpp stream / WhisperLive / RealtimeSTT / VoiceInk, FunASR streaming pipelines,
plus local inspection of Wispr Flow bundle resources (opus chunk streaming to cloud ASR).

I have enough grounding in both the findings and Veery's actual code. Here is the synthesized design recommendation.

---

# Veery Streaming Dictation — Design Recommendation

## 0. Grounding in Veery's current code (verified)

- `audio.py::AudioRecorder._process_vad_chunk` (line 406) already runs a per-32ms-frame VAD state machine: `WAITING → SPEECH_DETECTED → SILENCE_COUNTING → DONE`, appending 512-sample chunks to `self._buffer`. `SILENCE_COUNTING` counts silence frames and fires `DONE` at `vad_cfg.silence_duration_sec` (2.0s). This is the exact signal a segment-finalize design needs — it is already computing "speech just ended."
- `app.py::_stop_recording` (line 1128) calls `recorder.stop_and_flush()` → one `AudioSegment` → `_set_state(PROCESSING)` → spawns `_process_segment` worker + `_watch_processing` watchdog keyed on a `_processing_generation` counter with `processing_timeout_sec`.
- `_process_segment` (line 1236) pipeline: recorder VAD speech check → short/low-RMS energy gate → `stt.transcribe(audio, sr)` → generation-cancellation check → `_is_repetitive_hallucination` → `corrector.correct` → accessibility check → paste.
- `WhisperSTT.transcribe` (stt.py:646) writes a fresh temp WAV per call, uses `condition_on_previous_text=False`, `initial_prompt` from jargon via `_build_initial_prompt`, and has its own `_has_enough_speech_energy` gate. **No word_timestamps, no clip_timestamps today** — this matters below.

---

## 1. Recommended architecture: **VAD-boundary finalize + offset advance (Architecture B)**

Adopt the whisper.cpp-stream / WhisperLive pattern: **cut the audio at Silero VAD silence pauses, hand each completed segment to a single serialized background transcription worker while capture continues, and advance a sample cursor so finalized audio is never re-decoded.** On key release, transcribe only the residual tail since the last cut, splice, correct once, paste.

**Why this over the three alternatives in the findings:**

- **vs. LocalAgreement-2 / UFAL whisper_streaming (re-decode whole buffer every ~1s, commit longest agreeing prefix):** Rejected as primary. It re-encodes the entire growing buffer every step (O(n²) on stock mlx-whisper, which has no KV-cache reuse), *requires* `word_timestamps=True` — which (a) is a known ~10MB/call memory leak in mlx-whisper (mlx-examples #1254) amplified across N re-decodes, and (b) needs `sep=''` handling for spaceless Chinese that the shipped MLXWhisper adapter gets wrong (`sep=' '`). LocalAgreement's entire purpose is to recover word-safe cut points from *blind timer* chunking. **Veery already has word-safe cut points for free from Silero VAD**, so the expensive re-decode+agreement machinery buys nothing here. This is the single biggest "do not build" (see §6).
- **vs. WhisperKit eager mode:** Not portable. Its speed comes from CoreML/ANE graph edits (block-diagonal 15s attention mask, stateful KV-cache, silence caching) + a Swift LocalAgreement layer. Cannot be lifted into a Python/mlx-whisper app without re-exporting the model and adding a Swift subprocess dependency. Contradicts the mlx-only constraint.
- **vs. RealtimeSTT (Architecture A):** Explicitly rejected — its final pass on stop is still a *whole-clip* transcription, so it delivers live interim text but **zero release-latency saving**, which is Veery's actual goal.
- **vs. FunASR paraformer/SenseVoice second engine:** Rejected — drags in torch+torchaudio+onnx (~2GB, CPU-only on Apple Silicon), doubles resident memory alongside mlx-whisper, contends for the thermal budget, has weaker CN/EN code-switch than whisper-large-v3-turbo, and jargon `initial_prompt` biasing does not transfer (FunASR uses WFST hotwords). Its *2pass architecture* (cheap partials + authoritative offline pass at VAD endpoint) is worth stealing — but implement it with **chunked mlx-whisper for both passes**, not a second framework.

**Net:** one engine (existing mlx-whisper large-v3-turbo), one serialized worker, driven by the VAD boundaries Veery already computes. Release-to-text latency collapses to ~the final tail segment.

---

## 2. Segmentation policy

Decouple the **mid-clip finalize pause** from the **end-of-recording pause**. Veery's `silence_duration_sec=2.0` is deliberately long to avoid cutting accented speakers mid-utterance — keep that as the *recording-end* threshold, but introduce a *shorter* internal finalize threshold.

| Parameter | Start value | Rationale |
|---|---|---|
| `stream_finalize_pause_sec` | **0.7** | Between RealtimeSTT's 0.6 and whisper.cpp's ~1.0s trailing silence. Cuts a segment when the VAD has counted this much continuous silence *without* yet hitting the 2.0s recording-end. |
| `stream_min_segment_sec` | **1.0** | Whisper accuracy degrades badly <1s. Segments shorter than this are **not** finalized standalone — merge forward (keep accumulating into the next segment). |
| `stream_max_segment_sec` | **18** (cap in 15–30 range) | Force-finalize a monologue with no pause so context/memory don't blow up. WhisperLive uses 30–45s buffer; large-v3-turbo is happy well below that. |
| `stream_overlap_ms` | **200** | Prepend the last 200ms of the *previous* segment's audio to each finalized segment (whisper.cpp `keep_ms=200`) so a word straddling the cut isn't truncated. Because cuts land on VAD *silence*, this overlap is mostly silence, not repeated words. |
| VAD `silence_duration_sec` | **2.0 (unchanged)** | Still the recording-end signal. |

**Cut trigger (concrete):** inside `_process_vad_chunk`'s `SILENCE_COUNTING` branch, when accumulated `silence_sec` crosses `stream_finalize_pause_sec` **but is still below** `silence_duration_sec`, and the accumulated speech since the last cut ≥ `stream_min_segment_sec`, emit a **finalize event** for `_buffer[last_cut : now]` and record a `last_finalized_sample` cursor. Do **not** clear `_buffer` (the raw/VAD buffers stay intact for the existing manual-stop fallback and for the final splice) — just advance the cursor. Also fire a finalize when `now - last_cut ≥ stream_max_segment_sec` regardless of silence.

Keep the 512-sample/32ms Silero chunk invariant untouched — the finalize logic reads counters that already exist, it does not change chunk sizing.

---

## 3. Transcription continuity

- **Jargon prompt: include on every segment.** Reuse the existing `_build_whisper_jargon_prompt` output (already fed via `set_runtime_hints` → `_build_initial_prompt`). Whisper's `initial_prompt` is stateless per call, so each segment decode must carry it or jargon correction regresses mid-clip.
- **Rolling context: append the tail of confirmed text, capped.** Compose `initial_prompt = jargon_prompt + " " + last_120_chars_of_committed_transcript` (respect ~200-char / prompt-token budget from the findings). This preserves capitalization and CN/EN code-switch continuity across joins, mirroring whisper.cpp `prompt_tokens`.
  - **Caveat (important):** `condition_on_previous_text=False` stays as-is inside `transcribe`. Feeding rolling text via `initial_prompt` is a *known hallucination-loop trigger* on accented speech + long silences. Mitigate by (a) capping to ~120 chars, (b) only appending committed (already-VAD-validated) text, and (c) the repeat guard in §5. If Phase-1 testing shows drift, **drop the rolling tail and keep only the jargon prompt** — segments are already word-safe from VAD cuts, so cross-segment context is a nice-to-have, not load-bearing.
- **Splicing:** plain concatenation of each segment's `text`. Whisper's tokenizer emits a leading space for English tokens and none for CJK, so naive concatenation yields correct spacing for code-switched CN/EN (verified pattern from WhisperLive). Add a thin dedup: if a segment's first token duplicates the previous segment's trailing token (overlap artifact), drop the duplicate; collapse doubled sentence-final punctuation (`。。` / `..`).
- **Run the corrector ONCE on the fully merged text**, not per-segment (matches current `_process_segment` calling `corrector.correct` once). Per-segment correction risks partial-context mis-corrections and makes learner signals noisy.

---

## 4. Key-release handling & integration with `_stop_recording` / watchdog

**What stays:** `_process_segment`'s VAD speech check, energy gate, cancellation check, `_is_repetitive_hallucination`, corrector, accessibility check, paste. The IDLE/RECORDING/PROCESSING enum stays. The generation-counter watchdog stays.

**What changes:**

1. **During RECORDING**, finalize events (from §2) push `(segment_audio, seq)` onto a single-worker queue. The worker runs the *same* `stt.transcribe → hallucination filter → (no corrector yet)` steps and appends validated raw text to an ordered `_committed_segments` list under a lock. **No paste happens during RECORDING** — committed text is rendered in the overlay pill only (live preview), not typed. (Partial paste into the target app is explicitly out of scope — too risky with revisions; see §6.)
2. **On key release, `_stop_recording`:**
   - Set `_recording_finalizing = True` (existing), stop the stream, `stop_and_flush()` → this returns the *residual tail* since `last_finalized_sample` (change `_build_segment`/`stop_and_flush` to slice from the cursor, or have `_stop_recording` slice the returned segment).
   - **Drain the queue:** wait (bounded, e.g. 3s) for any in-flight segment worker to finish so committed order is complete. If the worker is mid-inference, join it; if it hangs past the drain timeout, proceed with what's committed (data-loss guard: committed text is already durably in `_committed_segments`).
   - Transition to `PROCESSING`, bump `_processing_generation`, spawn `_process_segment` **for the tail only**. The tail is short → this is the latency win.
   - `_process_segment` then splices `"".join(_committed_segments) + tail_text`, runs the corrector once, pastes.
3. **Watchdog:** `_watch_processing` still guards the *final tail* pass (short, so `processing_timeout_sec` is generous). Give the **in-RECORDING segment worker its own lightweight per-segment timeout** (e.g. `stream_segment_timeout_sec ≈ 8`) so a stalled mid-clip decode drops *that segment* and logs, rather than wedging capture or tripping the main watchdog. The main watchdog must **not** fire during long RECORDING — it only starts at release (unchanged: it's spawned by `_stop_recording`). Key the mid-clip worker's health off "last segment completed" progress, not a single entry timestamp (per findings).

**Edge case — empty commit:** if no segment was finalized before release (short utterance < `stream_min_segment_sec`), the tail *is* the whole clip → behavior is identical to today's batch path. This is the automatic fallback and should be the config-flag-off behavior too.

---

## 5. Failure modes to guard

| Failure | Guard |
|---|---|
| **Hallucination on short/noisy segments** | Reuse existing `_has_enough_speech_energy` (in `transcribe`) + `_is_repetitive_hallucination` **per segment** before committing. `stream_min_segment_sec=1.0` merge rule keeps sub-1s fragments out. |
| **Runaway repeat from rolling prompt** | Adopt WhisperLive's `same_output_threshold`: if a force-finalized (max-cap) segment's text is byte-identical to the prior segment, drop it. If drift appears, disable rolling-tail prompt (§3 caveat). |
| **Out-of-order completion** | Single **serialized** worker + monotonic `seq` per segment; `_committed_segments` is an ordered list indexed by `seq`. mlx-whisper/Metal inference is not safely reentrant — never run two decodes concurrently. |
| **Worker still busy at release** | Bounded drain (join with `stream_drain_timeout_sec ≈ 3s`). Committed text is persisted before each inference, so a hung worker loses at most the one in-flight segment, never prior committed text. Release path is **idempotent** (paste happens exactly once, in the final `_process_segment`). |
| **User releases mid-word** | The 200ms overlap + the fact that the tail is decoded as one contiguous clip (last cut → release) means the final word is fully present in the tail decode. No word is split across the tail boundary because the tail is never itself cut. |
| **Segment worker dies / device dropout mid-clip** | `_committed_segments` durable; `stop_and_flush` already survives PortAudio errors (audio.py `_close_stream` never raises). Final splice uses whatever committed + whatever tail exists. |
| **mlx-whisper temp-wav multiplication** | Accept higher *total* compute (N short decodes + overlap) as the cost of lower *wall-clock-at-release*. Benchmark on battery/thermal-throttled M-series; if contention raises perceived latency, raise `stream_finalize_pause_sec` (fewer, longer segments). |

---

## 6. What NOT to build (explicit rejections)

1. **Do NOT implement LocalAgreement-2 re-decode-and-agree.** VAD-chunking already gives word-safe cut points; the whole-buffer re-decode is redundant O(n²) work and forces `word_timestamps=True` into the mlx-whisper leak (#1254) and `sep` CJK breakage. This is the primary over-engineering trap in the findings.
2. **Do NOT add FunASR / paraformer / streaming-SenseVoice as a second engine.** ~2GB torch stack, CPU-only, memory doubling, weaker code-switch, no jargon-prompt transfer. Steal only the 2pass *architecture*, implemented with mlx-whisper.
3. **Do NOT use `word_timestamps=True` or `clip_timestamps`.** The leak (#1254) and the hang-when-end==duration (#1256) / decode-slowdown (#1285) bugs. Architecture B needs neither — it slices the raw float array at VAD-frame boundaries (which we already track in samples), not via Whisper timestamps.
4. **Do NOT paste partial/committed text into the target app during RECORDING.** Show it in the overlay pill only. Streaming partials get revised; typing-then-correcting into the user's editor causes visible churn and corrupts the learner's manual-edit detection.
5. **Do NOT port WhisperKit** (Swift/CoreML, non-portable).
6. **Do NOT switch to blind fixed-interval chunking** (simonw gist anti-pattern) — cuts words, loses context.

---

## 7. Phased implementation plan

### Phase 1 — Minimal viable streaming (behind `config.stt.streaming_enabled`, default off)

**`config.py`** — add `StreamingConfig` (frozen dataclass) with defaults:
```
streaming_enabled=False
stream_finalize_pause_sec=0.7
stream_min_segment_sec=1.0
stream_max_segment_sec=18.0
stream_overlap_ms=200
stream_segment_timeout_sec=8.0
stream_drain_timeout_sec=3.0
stream_rolling_prompt_chars=120   # 0 disables rolling tail
```

**`audio.py::AudioRecorder`:**
- Add a `last_finalized_sample` cursor + a `finalize_callback` (set by app).
- In `_process_vad_chunk` `SILENCE_COUNTING` branch: when `stream_finalize_pause_sec ≤ silence_sec < silence_duration_sec` and accumulated speech since cursor ≥ `stream_min_segment_sec`, snapshot `_buffer[cursor:]` (with `stream_overlap_ms` pre-roll from before the cursor), advance `last_finalized_sample`, and invoke `finalize_callback(audio_slice, seq)` **outside the RT-critical path** (enqueue only — keep the callback allocation-free; do the `np.concatenate` on the worker thread). Add the max-cap finalize.
- Make `stop_and_flush`/`_build_segment` return only the **tail** (`_buffer` from `last_finalized_sample` onward) when streaming is enabled; full buffer when disabled.

**`app.py`:**
- New `_StreamingSession` helper: single `queue.Queue`, one daemon worker thread, ordered `_committed_segments`, a lock. Worker runs `stt.transcribe` + energy/hallucination filters (reuse existing functions), appends validated text.
- `finalize_callback` (registered on the recorder in `_begin_recording`/`_start_recording`) enqueues segments.
- `_stop_recording`: when streaming enabled → drain queue (bounded by `stream_drain_timeout_sec`), then spawn `_process_segment` for the **tail**; `_process_segment` splices `_committed_segments + tail`, corrects once, pastes. When disabled → current path unchanged.
- Overlay: render committed text as live preview (optional in Phase 1 — can start with just the existing "recording" pill).
- Watchdog: unchanged for the tail pass; add the per-segment timeout in the streaming worker.

**Tests:** mock `stt.transcribe` to return per-segment stubs; assert (a) N-segment dictation splices in order, (b) sub-1s fragments merge, (c) release with empty commit == batch behavior, (d) hung segment worker doesn't lose committed text, (e) CN/EN concatenation spacing. Keep the existing real-VAD integration test.

### Phase 2 — Polish

- Live overlay preview with `incremental_text` diff (only paint new suffix; from streaming-sensevoice pattern).
- Rolling-tail `initial_prompt` (§3) with A/B measurement of hallucination rate; keep off if it regresses accented speech.
- `same_output_threshold`-style repeat guard on max-cap force-finalized segments.
- Adaptive `stream_finalize_pause_sec` based on measured decode latency vs. real-time (raise it under thermal throttling to reduce segment count).
- Punctuation-dedup + overlap-token-dedup refinement across splices.
- Metrics: log per-dictation "release-to-paste" latency and total decode count to validate the win on target hardware.

**Relevant files:** `src/veery/config.py` (new `StreamingConfig`), `src/veery/audio.py` (`_process_vad_chunk`, `_build_segment`/`stop_and_flush`, cursor + finalize callback), `src/veery/app.py` (`_begin_recording`/`_start_recording` to register callback, `_stop_recording`, `_process_segment` to splice, new `_StreamingSession`), `src/veery/stt.py` (`_build_initial_prompt` to accept an optional rolling-tail arg — Phase 2 only). `corrector.py`/`jargon.py`/`output.py` unchanged.

## Primary sources
- https://github.com/ufal/whisper_streaming
- https://raw.githubusercontent.com/ufal/whisper_streaming/main/whisper_online.py
- https://arxiv.org/html/2307.14743v2
- https://arxiv.org/abs/2307.14743
- https://aclanthology.org/2023.ijcnlp-demo.3.pdf
- https://arxiv.org/html/2507.10860v1 — WhisperKit On-device Real-time ASR paper (LocalAgreement policy, hypothesis vs confirmed streams, 0.45s/1.7s per-word latency, block-diagonal d750 mask, encoder 602ms to 218ms M3 Max, decoder 4.6ms, silence caching; model changes not pure orchestration)
- https://github.com/argmaxinc/WhisperKit/issues/102 — Eager streaming = predict same token >=2x; speculative decoding proposal (distil-large-v3 draft + large-v3 oracle, shared AudioEncoder)
- https://github.com/argmaxinc/WhisperKit/issues/111 — English text normalization for eager streaming mode confirmation
- https://github.com/ufal/whisper_streaming — LocalAgreement-2 backend-agnostic orchestration with an mlx-whisper backend; min-chunk-size, segment vs sentence trimming, init-prompt context passing (directly reusable)
- https://github.com/ml-explore/mlx-examples/issues/1258 — request for mlx_whisper realtime; naive 5s chunking cuts words
- https://github.com/ml-explore/mlx-examples/issues/1254 — memory grows ~10MB/call with word_timestamps=True (find_alignment/forward_with_cross_qk), unresolved
- https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp
- https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/README.md
- https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/audio_recorder.py
- https://github.com/KoljaB/RealtimeSTT/blob/master/docs/configuration.md
- https://github.com/KoljaB/RealtimeSTT/blob/master/README.md
- https://github.com/collabora/WhisperLive/blob/main/whisper_live/server.py
- https://github.com/modelscope/FunASR
- https://huggingface.co/funasr/paraformer-zh-streaming
- https://github.com/modelscope/FunASR/blob/main/runtime/docs/websocket_protocol.md
- https://github.com/modelscope/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online.md
- https://github.com/FunAudioLLM/SenseVoice
- https://github.com/pengzhendong/streaming-sensevoice
