"""Audio capture with Silero VAD for speech endpoint detection.

Uses sounddevice.InputStream with a real-time callback to capture 16kHz mono audio.
The callback rechunks variable-size PortAudio blocks into exact 32ms (512-sample)
frames for Silero VAD v5 (via torch.hub), which classifies each chunk as speech/silence.
A state machine tracks speech onset and offset to produce complete utterances.
"""

from __future__ import annotations

import enum
import logging
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch

from veery.config import AudioConfig, VADConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """A captured audio segment ready for STT."""

    audio: np.ndarray  # float32, mono, shape (n_samples,)
    sample_rate: int
    duration_sec: float


class _State(enum.Enum):
    """VAD state machine states."""

    WAITING = "waiting"  # No speech detected yet
    SPEECH_DETECTED = "speech_detected"  # Speech is happening
    SILENCE_COUNTING = "silence_counting"  # Speech stopped, counting silence frames
    DONE = "done"  # Enough silence after speech, recording complete


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

        # Rechunk buffer: accumulates incoming audio into exact VAD-sized chunks
        self._rechunk_buf = np.zeros(self._audio_cfg.chunk_samples, dtype=np.float32)
        self._rechunk_pos = 0

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

    def prepare_stream(self) -> None:
        """Open and start the audio stream immediately (call from hotkey thread).

        Audio chunks are buffered via the callback even before start_recording()
        finalizes state. This eliminates stream startup latency from the
        user-perceived recording delay.
        """
        if self._stream is not None:
            return  # already open

        with self._lock:
            # Reset state for a new recording
            max_chunks = int(self._audio_cfg.max_duration_sec * 1000 / self._audio_cfg.chunk_duration_ms)
            self._buffer = deque(maxlen=max_chunks)
            self._raw_buffer = deque(maxlen=max_chunks)

            # Pre-speech buffer: ~500ms worth of chunks to capture speech onset
            pre_speech_chunks = max(1, 500 // self._audio_cfg.chunk_duration_ms)
            self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)

            self._state = _State.WAITING
            self._speech_frames = 0
            self._silence_frames = 0
            self._done_event.clear()

        # Reset VAD hidden state for a fresh sequence
        if self._vad_model is not None:
            self._vad_model.reset_states()

        # Reset rechunk buffer for the new recording
        self._rechunk_buf[:] = 0.0
        self._rechunk_pos = 0

        self._stream = sd.InputStream(
            samplerate=self._audio_cfg.sample_rate,
            channels=self._audio_cfg.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio stream opened and capturing.")

    def start_recording(self) -> None:
        """Finalize recording setup (stream should already be open via prepare_stream).

        If the stream isn't open yet (e.g. direct call), opens it as a fallback.
        """
        self._ensure_vad_loaded()

        if self._stream is None or not self._stream.active:
            self.prepare_stream()

        logger.info("Recording started.")

    def stop_recording(self) -> AudioSegment | None:
        """Stop the audio stream and return the captured segment.

        Returns None if no valid speech was detected (too short or no speech).
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Recording stopped.")

        # Unblock any thread waiting in wait_for_speech_end()
        self._done_event.set()

        return self._build_segment()

    def stop_and_flush(self) -> AudioSegment | None:
        """Stop recording and return all captured audio, using raw buffer as fallback.

        Use this for manual stops where the user explicitly ended recording.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Recording stopped.")

        self._done_event.set()
        return self._build_segment(use_raw=True)

    def wait_for_speech_end(self, timeout: float | None = None) -> AudioSegment | None:
        """Block until VAD detects end of speech, then return the segment.

        Args:
            timeout: Maximum seconds to wait. Defaults to max_duration_sec.

        Returns:
            AudioSegment if valid speech was captured, None otherwise.
        """
        if timeout is None:
            timeout = self._audio_cfg.max_duration_sec

        self._done_event.wait(timeout=timeout)
        return self.stop_recording()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def state(self) -> _State:
        with self._lock:
            return self._state

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

        # Apply input gain if configured (for quiet microphones)
        if self._audio_cfg.input_gain != 1.0:
            np.multiply(mono, self._audio_cfg.input_gain, out=mono)
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

            if self._state == _State.DONE:
                if is_speech:
                    # Speech resumed after silence timeout — re-activate.
                    self._buffer.append(chunk)
                    self._silence_frames = 0
                    self._state = _State.SPEECH_DETECTED
                    self._speech_frames += 1
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

            elif self._state == _State.SPEECH_DETECTED:
                self._buffer.append(chunk)
                if is_speech:
                    self._speech_frames += 1
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
                else:
                    self._silence_frames += 1
                    silence_sec = self._silence_frames * self._audio_cfg.chunk_duration_ms / 1000
                    if silence_sec >= self._vad_cfg.silence_duration_sec:
                        self._state = _State.DONE
                        self._done_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
                # when it looks speech-like (enough duration + non-trivial energy).
                raw_audio = np.concatenate(list(self._raw_buffer))
                raw_duration_sec = len(raw_audio) / self._audio_cfg.sample_rate
                raw_rms = float(np.sqrt(np.mean(raw_audio**2)))
                raw_peak = float(np.max(np.abs(raw_audio)))

                min_manual_duration_sec = max(self._vad_cfg.min_speech_duration_sec, 0.25)
                rms_threshold = 0.01
                peak_threshold = 0.06
                if raw_duration_sec >= min_manual_duration_sec and (
                    raw_rms >= rms_threshold or raw_peak >= peak_threshold
                ):
                    source = self._raw_buffer
                    logger.info(
                        "Using raw-audio fallback (state=%s, duration=%.2fs, rms=%.4f, peak=%.4f).",
                        self._state.value,
                        raw_duration_sec,
                        raw_rms,
                        raw_peak,
                    )
                elif self._state == _State.WAITING:
                    logger.info(
                        "VAD detected no speech; raw fallback rejected "
                        "(duration=%.2fs, rms=%.4f, peak=%.4f).",
                        raw_duration_sec,
                        raw_rms,
                        raw_peak,
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
            )
