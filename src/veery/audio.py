"""Audio capture with Silero VAD for speech endpoint detection.

Uses sounddevice.InputStream with a real-time callback to capture 16kHz mono audio.
The callback rechunks variable-size PortAudio blocks into exact 32ms (512-sample)
frames for Silero VAD v5 (via torch.hub), which classifies each chunk as speech/silence.
A state machine tracks speech onset and offset to produce complete utterances.
"""

from __future__ import annotations

import enum
import itertools
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch

from veery.config import AudioConfig, StreamingConfig, VADConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """A captured audio segment ready for STT."""

    audio: np.ndarray  # float32, mono, shape (n_samples,)
    sample_rate: int
    duration_sec: float
    # True when the recording-time VAD confirmed speech in this segment —
    # lets the pipeline skip a costly second full-segment VAD pass.
    vad_confirmed: bool = False


class _State(enum.Enum):
    """VAD state machine states."""

    WAITING = "waiting"  # No speech detected yet
    SPEECH_DETECTED = "speech_detected"  # Speech is happening
    SILENCE_COUNTING = "silence_counting"  # Speech stopped, counting silence frames
    DONE = "done"  # Enough silence after speech, recording complete


class StopReason(enum.Enum):
    """Why the current recording session ended."""

    USER_STOP = "user_stop"
    MANUAL_CAP_REACHED = "manual_cap_reached"
    SPEECH_END = "speech_end"
    WAIT_TIMEOUT = "wait_timeout"


class _CaptureMode(enum.Enum):
    WAIT = "wait"
    MANUAL = "manual"


class AudioRecorder:
    """Microphone capture with Silero VAD-based speech endpoint detection.

    The audio stream is only open during active recording (battery-friendly).
    Thread-safe: a lock protects the audio buffer, an event signals completion.

    Usage::

        recorder = AudioRecorder(audio_config, vad_config)
        recorder.start_recording()
        segment = recorder.wait_for_speech_end(timeout=30)
        if segment is not None:
            # segment.audio is a float32 numpy array
            transcribe(segment)
    """

    def __init__(self, audio_config: AudioConfig, vad_config: VADConfig) -> None:
        self._audio_cfg = audio_config
        self._vad_cfg = vad_config

        # VAD model (loaded once, reused across recordings)
        self._vad_model: torch.nn.Module | None = None
        self._vad_lock = threading.Lock()

        # Recording state (protected by _lock)
        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        self._raw_buffer: deque[np.ndarray] = deque()  # all audio for manual stop
        self._pre_speech_buffer: deque[np.ndarray] = deque()
        self._state = _State.WAITING
        self._speech_frames = 0
        self._silence_frames = 0
        self._done_event = threading.Event()
        self._manual_stop_event = threading.Event()
        self._stop_reason: StopReason | None = None
        self._capture_mode = _CaptureMode.WAIT
        self._captured_samples = 0
        self._manual_max_samples: int | None = None

        # Rechunk buffer: accumulates incoming audio into exact VAD-sized chunks
        self._rechunk_buf = np.zeros(self._audio_cfg.chunk_samples, dtype=np.float32)
        self._rechunk_pos = 0

        # Streaming finalize state (see design/streaming-dictation.md).
        # Segments are cut at short VAD pauses and handed to the callback so
        # they can be transcribed while capture continues. All counters are in
        # VAD-chunk units and protected by self._lock.
        self._streaming_cfg: StreamingConfig | None = None
        self._finalize_cb: Callable[[int, list[np.ndarray], int], None] | None = None
        self._finalized_chunks = 0  # index into _buffer past the last cut
        self._segment_seq = 0
        self._speech_chunks_since_cut = 0
        self._finalize_pause_chunks = 0
        self._min_segment_speech_chunks = 0
        self._max_segment_chunks = 0
        self._overlap_chunks = 0

        # sounddevice stream (created per-recording)
        self._stream: sd.InputStream | None = None
        self._min_speech_frames = 3  # guard against single-frame VAD spikes (keyboard/noise)

    # ------------------------------------------------------------------
    # VAD model management
    # ------------------------------------------------------------------

    def _ensure_vad_loaded(self) -> torch.nn.Module:
        """Load Silero VAD model on first use (thread-safe)."""
        if self._vad_model is not None:
            return self._vad_model

        with self._vad_lock:
            # Double-checked locking
            if self._vad_model is not None:
                return self._vad_model

            logger.info("Loading Silero VAD model...")
            # Prefer cached model to avoid network dependency at startup.
            from pathlib import Path

            cache_dir = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
            if cache_dir.is_dir():
                try:
                    model, _ = torch.hub.load(
                        str(cache_dir),
                        "silero_vad",
                        source="local",
                        trust_repo=True,
                    )
                except Exception:
                    logger.warning("Local VAD cache failed, falling back to remote.")
                    model, _ = torch.hub.load(
                        "snakers4/silero-vad",
                        "silero_vad",
                        trust_repo=True,
                    )
            else:
                model, _ = torch.hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    trust_repo=True,
                )
            model.eval()
            self._vad_model = model
            logger.info("Silero VAD model loaded.")
            return self._vad_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure_streaming(self, cfg: StreamingConfig | None) -> None:
        """Set streaming segmentation thresholds (chunk units). None disables."""
        with self._lock:
            self._streaming_cfg = cfg if (cfg is not None and cfg.enabled) else None
            if self._streaming_cfg is None:
                return
            ms = self._audio_cfg.chunk_duration_ms
            self._finalize_pause_chunks = max(1, int(cfg.finalize_pause_sec * 1000 / ms))
            self._min_segment_speech_chunks = max(1, int(cfg.min_segment_sec * 1000 / ms))
            self._max_segment_chunks = max(2, int(cfg.max_segment_sec * 1000 / ms))
            self._overlap_chunks = max(0, int(cfg.overlap_ms / ms))

    def set_finalize_callback(
        self, callback: Callable[[int, list[np.ndarray], int], None] | None
    ) -> None:
        """Register the per-recording segment sink: callback(seq, chunks, end_chunk).

        Called from the real-time audio thread — the callback must only
        enqueue (no allocation-heavy work, no blocking).
        """
        with self._lock:
            self._finalize_cb = callback

    def prepare_stream(self, *, manual_mode: bool = False) -> None:
        """Open and start the audio stream immediately (call from hotkey thread).

        Audio chunks are buffered via the callback even before start_recording()
        finalizes state. This eliminates stream startup latency from the
        user-perceived recording delay.
        """
        if self._stream is not None:
            return  # already open

        with self._lock:
            # Reset state for a new recording.
            #
            # Manual hold/toggle recordings must keep the full capture until the
            # user stops. A bounded deque turns long dictations into a rolling
            # window and silently drops the beginning of the paragraph.
            self._buffer = deque()
            self._raw_buffer = deque()

            # Pre-speech buffer: ~500ms worth of chunks to capture speech onset
            pre_speech_chunks = max(1, 500 // self._audio_cfg.chunk_duration_ms)
            self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)

            self._state = _State.WAITING
            self._speech_frames = 0
            self._silence_frames = 0
            self._done_event.clear()
            self._manual_stop_event.clear()
            self._stop_reason = None
            self._capture_mode = _CaptureMode.MANUAL if manual_mode else _CaptureMode.WAIT
            self._captured_samples = 0
            self._finalized_chunks = 0
            self._segment_seq = 0
            self._speech_chunks_since_cut = 0

            manual_max_duration = self._audio_cfg.manual_max_duration_sec
            if manual_mode and manual_max_duration is not None:
                self._manual_max_samples = int(self._audio_cfg.sample_rate * manual_max_duration)
            else:
                self._manual_max_samples = None

        # Reset VAD hidden state for a fresh sequence
        if self._vad_model is not None:
            self._vad_model.reset_states()

        # Reset rechunk buffer for the new recording
        self._rechunk_buf[:] = 0.0
        self._rechunk_pos = 0

        stream = sd.InputStream(
            samplerate=self._audio_cfg.sample_rate,
            channels=self._audio_cfg.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        try:
            stream.start()
        except Exception:
            # A stream that was created but failed to start must not be kept,
            # or the early-return above would treat it as "already open" and
            # recording would silently never work again.
            try:
                stream.close()
            except Exception:
                logger.debug("Could not close unstarted stream", exc_info=True)
            raise
        self._stream = stream
        logger.info("Audio stream opened and capturing.")

    def start_recording(self, *, manual_mode: bool = False, open_stream_if_needed: bool = True) -> None:
        """Finalize recording setup (stream should already be open via prepare_stream).

        If the stream isn't open yet (e.g. direct call), opens it as a fallback.
        """
        self._ensure_vad_loaded()

        if open_stream_if_needed and (self._stream is None or not self._stream.active):
            self.prepare_stream(manual_mode=manual_mode)

        logger.info("Recording started.")

    def stop_recording(self, *, reason: StopReason | None = None) -> AudioSegment | None:
        """Stop the audio stream and return the captured segment.

        Returns None if no valid speech was detected (too short or no speech).
        """
        self._close_stream()
        self._set_stop_reason(reason or StopReason.WAIT_TIMEOUT)
        self._done_event.set()
        self._manual_stop_event.set()

        return self._build_segment()

    def stop_and_flush(self, *, reason: StopReason = StopReason.USER_STOP) -> AudioSegment | None:
        """Stop recording and return all captured audio, using raw buffer as fallback.

        Use this for manual stops where the user explicitly ended recording.
        The stream is detached synchronously but stopped/closed on a background
        thread: PortAudio's stop() can block for 100-300ms and none of that
        work is needed before transcription can start.
        """
        stream = self._stream
        self._stream = None
        if stream is not None:
            threading.Thread(
                target=self._close_detached_stream, args=(stream,), daemon=True
            ).start()
        self._set_stop_reason(reason)
        self._done_event.set()
        self._manual_stop_event.set()
        return self._build_segment(use_raw=True)

    def stop_capture(self, *, reason: StopReason = StopReason.USER_STOP) -> None:
        """Stop capturing without building a segment (streaming release path).

        Detaches the stream (closed on a background thread), clears the
        finalize callback so no further segments are emitted, and signals
        waiters. Buffers stay intact for build_tail().
        """
        stream = self._stream
        self._stream = None
        if stream is not None:
            threading.Thread(
                target=self._close_detached_stream, args=(stream,), daemon=True
            ).start()
        with self._lock:
            self._finalize_cb = None
        self._set_stop_reason(reason)
        self._done_event.set()
        self._manual_stop_event.set()

    def snapshot_capture(self) -> np.ndarray | None:
        """Full captured audio of the current/most recent recording.

        Used by the dictation-history archive; prefers the VAD buffer, falls
        back to the raw buffer (VAD-missed captures).
        """
        with self._lock:
            source = self._buffer if self._buffer else self._raw_buffer
            if not source:
                return None
            return np.concatenate(list(source))

    def build_tail(self, from_chunk: int) -> AudioSegment | None:
        """Build the residual segment after the last committed streaming cut.

        With from_chunk <= 0 (nothing committed) this is exactly the batch
        path — full segment with raw-audio fallback. Otherwise the tail is
        _buffer[from_chunk - overlap:], with trailing silence stripped.
        """
        if from_chunk <= 0:
            return self._build_segment(use_raw=True)

        with self._lock:
            start = max(0, from_chunk - self._overlap_chunks)
            chunks = list(itertools.islice(self._buffer, start, None))
            if not chunks:
                return None
            audio = np.concatenate(chunks)
            silence_samples = int(self._silence_frames * self._audio_cfg.chunk_samples)
            if 0 < silence_samples < len(audio):
                audio = audio[:-silence_samples]
            duration_sec = len(audio) / self._audio_cfg.sample_rate
            if duration_sec < 0.25:
                return None
            has_new_speech = self._speech_chunks_since_cut >= 3
            logger.info(
                "Built streaming tail: %.2fs (new speech: %s).", duration_sec, has_new_speech
            )
            return AudioSegment(
                audio=audio,
                sample_rate=self._audio_cfg.sample_rate,
                duration_sec=duration_sec,
                vad_confirmed=has_new_speech,
            )

    @staticmethod
    def _close_detached_stream(stream: sd.InputStream) -> None:
        try:
            try:
                stream.stop()
            finally:
                stream.close()
        except Exception:
            logger.exception("Error closing audio stream (device disconnected?)")
            return
        logger.info("Recording stopped.")

    def wait_for_speech_end(self, timeout: float | None = None) -> AudioSegment | None:
        """Block until VAD detects end of speech, then return the segment.

        Args:
            timeout: Maximum seconds to wait. Defaults to wait_timeout_sec.

        Returns:
            AudioSegment if valid speech was captured, None otherwise.
        """
        if timeout is None:
            timeout = self._audio_cfg.wait_timeout_sec

        completed = self._done_event.wait(timeout=timeout)
        reason = StopReason.SPEECH_END if completed else StopReason.WAIT_TIMEOUT
        return self.stop_recording(reason=reason)

    def wait_for_manual_stop(self, timeout: float | None = None) -> StopReason | None:
        """Block until a manual recording stop reason is available."""
        if not self._manual_stop_event.wait(timeout=timeout):
            return None

        with self._lock:
            return self._stop_reason

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def state(self) -> _State:
        with self._lock:
            return self._state

    @property
    def stop_reason(self) -> StopReason | None:
        with self._lock:
            return self._stop_reason

    def has_speech(self, audio: np.ndarray) -> bool:
        """Run a lightweight VAD pass on a captured segment.

        This is used as a second-stage gate before STT to avoid sending
        near-silent/manual-stop clips that can trigger Whisper hallucinations.

        Deliberately lenient: audio reaching this gate already passed either
        the recording-time VAD or the raw-fallback energy heuristic, so this
        pass only needs to reject true silence/noise. A stricter re-gate here
        would silently drop quiet or accented speech (VAD probabilities
        differ slightly between passes).
        """
        if audio.size < self._audio_cfg.chunk_samples:
            return False

        model = self._ensure_vad_loaded()
        try:
            model.reset_states()
        except Exception:
            pass

        # Slightly below the recording threshold: borderline speech that
        # triggered recording must not be rejected by a coin-flip re-run.
        threshold = max(0.2, self._vad_cfg.threshold - 0.1)

        speech_frames = 0
        max_consecutive = 0
        consecutive = 0
        total_frames = 0

        for start in range(0, audio.size - self._audio_cfg.chunk_samples + 1, self._audio_cfg.chunk_samples):
            chunk = audio[start : start + self._audio_cfg.chunk_samples]
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
            try:
                prob = model(chunk_tensor, self._audio_cfg.sample_rate).item()
            except Exception:
                prob = 0.0

            total_frames += 1
            if prob >= threshold:
                speech_frames += 1
                consecutive += 1
                if consecutive > max_consecutive:
                    max_consecutive = consecutive
            else:
                consecutive = 0

        try:
            model.reset_states()
        except Exception:
            pass

        if total_frames == 0:
            return False

        duration_sec = audio.size / self._audio_cfg.sample_rate
        # 2 consecutive frames = 64ms of sustained speech probability; the
        # old active-ratio gate penalized long holds with short utterances.
        min_frames = 2 if duration_sec < 1.0 else 3
        return speech_frames >= min_frames and max_consecutive >= 2

    # ------------------------------------------------------------------
    # sounddevice callback (runs on real-time audio thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio chunk.

        MUST be fast: no I/O, no logging. VAD inference is ~0.1ms on CPU.
        Rechunks variable-size PortAudio blocks into exact 512-sample
        (32ms) chunks required by Silero VAD.
        """
        if status:
            # Overflow or underflow — nothing we can do in real-time
            pass

        mono = indata[:, 0].copy()

        # Apply input gain if configured (for quiet microphones). The
        # configured gain is a maximum: the effective gain is capped per chunk
        # so a loud microphone is never driven into clipping distortion, while
        # quiet mics still get the full boost.
        if self._audio_cfg.input_gain != 1.0:
            # Scalar reductions only — np.abs() would allocate a temp array,
            # which the RT callback must not do.
            peak = float(max(mono.max(), -mono.min()))
            gain = self._audio_cfg.input_gain
            if peak > 1e-6:
                gain = min(gain, 0.98 / peak)
            if gain > 1.0:
                np.multiply(mono, gain, out=mono)
            np.clip(mono, -1.0, 1.0, out=mono)  # Clamp to valid audio range

        vad_size = self._audio_cfg.chunk_samples
        pos = 0

        while pos < len(mono):
            space = vad_size - self._rechunk_pos
            n = min(space, len(mono) - pos)
            self._rechunk_buf[self._rechunk_pos : self._rechunk_pos + n] = mono[pos : pos + n]
            self._rechunk_pos += n
            pos += n

            if self._rechunk_pos >= vad_size:
                self._process_vad_chunk(self._rechunk_buf.copy())
                self._rechunk_pos = 0

    def _process_vad_chunk(self, chunk: np.ndarray) -> None:
        """Process a single VAD-sized audio chunk through the state machine.

        Called from _audio_callback for each complete 512-sample chunk.
        """
        try:
            chunk_tensor = torch.from_numpy(chunk)
            speech_prob = self._vad_model(chunk_tensor, self._audio_cfg.sample_rate).item()
        except Exception:
            # Cannot log from RT callback; treat failed frame as silence.
            speech_prob = 0.0

        is_speech = speech_prob >= self._vad_cfg.threshold

        with self._lock:
            # Always keep raw audio for manual stop fallback
            self._raw_buffer.append(chunk)
            self._captured_samples += chunk.size
            if (
                self._capture_mode == _CaptureMode.MANUAL
                and self._manual_max_samples is not None
                and self._captured_samples >= self._manual_max_samples
                and self._stop_reason is None
            ):
                self._stop_reason = StopReason.MANUAL_CAP_REACHED
                self._manual_stop_event.set()

            if self._state == _State.DONE:
                if is_speech:
                    # Speech resumed after silence timeout — re-activate.
                    self._buffer.append(chunk)
                    self._silence_frames = 0
                    self._state = _State.SPEECH_DETECTED
                    self._speech_frames += 1
                    self._speech_chunks_since_cut += 1
                    self._done_event.clear()
                return

            if self._state == _State.WAITING:
                # Accumulate pre-speech ring buffer
                self._pre_speech_buffer.append(chunk)
                if is_speech:
                    # Flush pre-speech buffer into main buffer
                    for pre_chunk in self._pre_speech_buffer:
                        self._buffer.append(pre_chunk)
                    self._pre_speech_buffer.clear()
                    self._state = _State.SPEECH_DETECTED
                    self._speech_frames = 1
                    self._speech_chunks_since_cut += 1

            elif self._state == _State.SPEECH_DETECTED:
                self._buffer.append(chunk)
                if is_speech:
                    self._speech_frames += 1
                    self._speech_chunks_since_cut += 1
                    # Force-cut a no-pause monologue so segments stay short
                    # enough to decode well (and start decoding early).
                    if (
                        self._streaming_active_locked()
                        and len(self._buffer) - self._finalized_chunks >= self._max_segment_chunks
                        and self._speech_chunks_since_cut >= self._min_segment_speech_chunks
                    ):
                        self._emit_segment_locked()
                else:
                    self._silence_frames = 1
                    self._state = _State.SILENCE_COUNTING

            elif self._state == _State.SILENCE_COUNTING:
                self._buffer.append(chunk)
                if is_speech:
                    # Speech resumed
                    self._silence_frames = 0
                    self._state = _State.SPEECH_DETECTED
                    self._speech_frames += 1
                    self._speech_chunks_since_cut += 1
                else:
                    self._silence_frames += 1
                    # Mid-clip streaming cut: a short pause (well before the
                    # recording-end threshold) finalizes the current segment
                    # for background transcription. Equality fires exactly
                    # once per silence run.
                    if (
                        self._streaming_active_locked()
                        and self._silence_frames == self._finalize_pause_chunks
                        and self._speech_chunks_since_cut >= self._min_segment_speech_chunks
                    ):
                        self._emit_segment_locked()
                    silence_sec = self._silence_frames * self._audio_cfg.chunk_duration_ms / 1000
                    if silence_sec >= self._vad_cfg.silence_duration_sec:
                        self._state = _State.DONE
                        self._done_event.set()

    def _streaming_active_locked(self) -> bool:
        return (
            self._streaming_cfg is not None
            and self._finalize_cb is not None
            and self._capture_mode == _CaptureMode.MANUAL
        )

    def _emit_segment_locked(self) -> None:
        """Finalize _buffer[cursor:] as a segment and advance the cursor.

        Runs on the RT audio thread under self._lock: only pointer copies
        (no audio concatenation) and a queue put in the callback — and it
        must never raise into the audio callback.
        """
        cb = self._finalize_cb
        end = len(self._buffer)
        if cb is None or end <= self._finalized_chunks:
            return
        start = max(0, self._finalized_chunks - self._overlap_chunks)
        chunks = list(itertools.islice(self._buffer, start, end))
        self._finalized_chunks = end
        self._speech_chunks_since_cut = 0
        seq = self._segment_seq
        self._segment_seq += 1
        try:
            cb(seq, chunks, end)
        except Exception:
            pass  # RT thread: no logging, no raising

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_stream(self) -> None:
        """Stop and close the current stream if it is active.

        Never raises: PortAudio can throw when the input device disappeared
        mid-recording (Bluetooth/USB mic dropout), and callers must still be
        able to flush the captured audio and reset state.
        """
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            try:
                stream.stop()
            finally:
                # close() must run even when stop() raises, or the PortAudio
                # stream object leaks after a device dropout.
                stream.close()
        except Exception:
            logger.exception("Error closing audio stream (device disconnected?)")
            return
        logger.info("Recording stopped.")

    def _set_stop_reason(self, reason: StopReason) -> None:
        """Record the first stop reason for this session."""
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason

    def _build_segment(self, use_raw: bool = False) -> AudioSegment | None:
        """Assemble buffered chunks into an AudioSegment.

        Args:
            use_raw: If True, use raw buffer as fallback when VAD buffer is empty.

        Returns None if speech was too short or never detected.
        """
        with self._lock:
            source = self._buffer
            if not source and use_raw and self._raw_buffer:
                # Manual stop: if VAD misses onset, fall back to raw audio only
                # when it looks speech-like (enough duration + sustained energy).
                raw_audio = np.concatenate(list(self._raw_buffer))
                raw_duration_sec = len(raw_audio) / self._audio_cfg.sample_rate
                raw_rms = float(np.sqrt(np.mean(raw_audio**2)))
                raw_peak = float(np.max(np.abs(raw_audio)))
                frame_size = max(1, int(self._audio_cfg.sample_rate * 0.02))  # 20ms
                n_frames = raw_audio.size // frame_size
                active_frames = 0
                active_ratio = 0.0
                if n_frames > 0:
                    framed = raw_audio[: n_frames * frame_size].reshape(n_frames, frame_size)
                    frame_rms = np.sqrt(np.mean(framed**2, axis=1))
                    active_frames = int(np.count_nonzero(frame_rms >= 0.015))
                    active_ratio = active_frames / n_frames

                # Gate on the ABSOLUTE amount of speech-like audio, not a
                # ratio over the whole hold: a 1s utterance in a 10s hold is
                # real speech even though its whole-hold ratio/RMS are tiny.
                min_manual_duration_sec = max(self._vad_cfg.min_speech_duration_sec, 0.25)
                if (
                    raw_duration_sec >= min_manual_duration_sec
                    and active_frames >= 8  # >=0.16s of energetic audio
                ):
                    source = self._raw_buffer
                    logger.info(
                        "Using raw-audio fallback (state=%s, duration=%.2fs, rms=%.4f, peak=%.4f, "
                        "active_frames=%d, active_ratio=%.2f).",
                        self._state.value,
                        raw_duration_sec,
                        raw_rms,
                        raw_peak,
                        active_frames,
                        active_ratio,
                    )
                elif self._state == _State.WAITING:
                    logger.info(
                        "VAD detected no speech; raw fallback rejected "
                        "(duration=%.2fs, rms=%.4f, peak=%.4f, active_frames=%d, active_ratio=%.2f).",
                        raw_duration_sec,
                        raw_rms,
                        raw_peak,
                        active_frames,
                        active_ratio,
                    )
            if not source:
                logger.debug("No audio in buffer.")
                return None

            audio = np.concatenate(list(source))
            duration_sec = len(audio) / self._audio_cfg.sample_rate

            # Strip trailing silence from the segment (only for VAD-segmented audio)
            if source is self._buffer:
                if self._speech_frames < self._min_speech_frames:
                    logger.debug(
                        "Discarding segment with too few speech frames (%d < %d).",
                        self._speech_frames,
                        self._min_speech_frames,
                    )
                    return None
                silence_samples = int(self._silence_frames * self._audio_cfg.chunk_samples)
                if silence_samples > 0 and silence_samples < len(audio):
                    audio = audio[:-silence_samples]
                    duration_sec = len(audio) / self._audio_cfg.sample_rate

            min_dur = self._vad_cfg.min_speech_duration_sec
            if duration_sec < min_dur:
                logger.debug(
                    "Speech too short (%.2fs < %.2fs), discarding.",
                    duration_sec,
                    min_dur,
                )
                return None

            logger.info("Captured %.2fs of audio.", duration_sec)
            return AudioSegment(
                audio=audio,
                sample_rate=self._audio_cfg.sample_rate,
                duration_sec=duration_sec,
                # The VAD buffer only fills after the recording-time VAD
                # confirmed speech; the raw fallback was only energy-gated.
                vad_confirmed=source is self._buffer,
            )
