"""Tests for AudioRecorder: VAD state machine, buffers, and stream lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from veery.audio import AudioRecorder, AudioSegment, _State
from veery.config import AudioConfig, VADConfig


def _make_recorder(
    audio_cfg: AudioConfig | None = None,
    vad_cfg: VADConfig | None = None,
) -> AudioRecorder:
    """Create an AudioRecorder with mocked sounddevice to avoid real mic access."""
    a = audio_cfg or AudioConfig()
    v = vad_cfg or VADConfig()
    with patch("veery.audio.sd"):
        recorder = AudioRecorder(a, v)
    return recorder


def _chunk(value: float = 0.0, samples: int = 512) -> np.ndarray:
    """Simulate an indata array from sounddevice (shape: (samples, 1))."""
    return np.full((samples, 1), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# AudioSegment dataclass
# ---------------------------------------------------------------------------


class TestAudioSegment:
    def test_fields(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        seg = AudioSegment(audio=audio, sample_rate=16000, duration_sec=1.0)
        assert seg.sample_rate == 16000
        assert seg.duration_sec == 1.0
        assert len(seg.audio) == 16000


# ---------------------------------------------------------------------------
# VAD state machine (via _audio_callback)
# ---------------------------------------------------------------------------


class TestVADStateMachine:
    def test_initial_state_is_waiting(self) -> None:
        rec = _make_recorder()
        assert rec.state == _State.WAITING

    def test_speech_transitions_to_speech_detected(self) -> None:
        rec = _make_recorder()
        rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.9)))

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))

        assert rec.state == _State.SPEECH_DETECTED
        assert rec._speech_frames == 1

    def test_silence_stays_waiting(self) -> None:
        rec = _make_recorder()
        rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.1)))

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))

        assert rec.state == _State.WAITING

    def test_speech_then_silence_transitions_to_silence_counting(self) -> None:
        rec = _make_recorder()
        speech_prob = iter([0.9, 0.1])
        rec._vad_model = MagicMock(
            return_value=MagicMock(item=MagicMock(side_effect=lambda: next(speech_prob)))
        )

        # First chunk: speech
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        assert rec.state == _State.SPEECH_DETECTED

        # Second chunk: silence
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        assert rec.state == _State.SILENCE_COUNTING
        assert rec._silence_frames == 1

    def test_silence_counting_resumes_on_speech(self) -> None:
        rec = _make_recorder()
        probs = iter([0.9, 0.1, 0.9])
        rec._vad_model = MagicMock(
            return_value=MagicMock(item=MagicMock(side_effect=lambda: next(probs)))
        )

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # speech
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # silence
        assert rec.state == _State.SILENCE_COUNTING

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # speech again
        assert rec.state == _State.SPEECH_DETECTED
        assert rec._silence_frames == 0

    def test_enough_silence_transitions_to_done(self) -> None:
        """After enough silence frames, state goes to DONE and event is set."""
        vad_cfg = VADConfig(silence_duration_sec=0.064)  # 2 chunks at 32ms each
        rec = _make_recorder(vad_cfg=vad_cfg)

        probs = iter([0.9, 0.1, 0.1])
        rec._vad_model = MagicMock(
            return_value=MagicMock(item=MagicMock(side_effect=lambda: next(probs)))
        )

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # speech
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # silence 1
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))  # silence 2

        assert rec.state == _State.DONE
        assert rec._done_event.is_set()

    def test_done_state_reactivates_on_speech(self) -> None:
        """Speech after silence timeout re-activates VAD to SPEECH_DETECTED."""
        rec = _make_recorder()
        rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.9)))

        with rec._lock:
            rec._state = _State.DONE
        rec._done_event.set()

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        assert rec.state == _State.SPEECH_DETECTED
        assert len(rec._buffer) == 1  # speech chunk appended
        assert len(rec._raw_buffer) == 1  # raw buffer always accumulates
        assert not rec._done_event.is_set()  # cleared on re-activation

    def test_done_state_accumulates_raw_on_silence(self) -> None:
        """Silence in DONE state still accumulates raw buffer."""
        rec = _make_recorder()
        rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.0)))

        with rec._lock:
            rec._state = _State.DONE

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        assert rec.state == _State.DONE  # stays DONE
        assert len(rec._buffer) == 0  # silence not appended to VAD buffer
        assert len(rec._raw_buffer) == 1  # raw buffer accumulates

    def test_speech_pause_speech_preserves_all_audio(self) -> None:
        """Speech → 2s silence → speech: all audio is captured in VAD buffer."""
        vad_cfg = VADConfig(silence_duration_sec=0.064)  # 2 chunks of silence = DONE
        rec = _make_recorder(vad_cfg=vad_cfg)

        # Sequence: speech, silence, silence (triggers DONE), silence, speech
        probs = iter([0.9, 0.1, 0.1, 0.1, 0.9])
        rec._vad_model = MagicMock(
            return_value=MagicMock(item=MagicMock(side_effect=lambda: next(probs)))
        )
        status = MagicMock(spec=False)

        rec._audio_callback(_chunk(0.5), 512, None, status)  # speech → SPEECH_DETECTED
        rec._audio_callback(_chunk(0.0), 512, None, status)  # silence → SILENCE_COUNTING
        rec._audio_callback(_chunk(0.0), 512, None, status)  # silence → DONE
        assert rec.state == _State.DONE

        rec._audio_callback(_chunk(0.0), 512, None, status)  # silence in DONE (raw only)
        assert rec.state == _State.DONE

        rec._audio_callback(_chunk(0.5), 512, None, status)  # speech → re-activates
        assert rec.state == _State.SPEECH_DETECTED

        # VAD buffer has: speech + 2 silence + resumed speech = 4 chunks
        # (the silence-in-DONE chunk is NOT in VAD buffer, only in raw)
        assert len(rec._buffer) == 4
        # Raw buffer has ALL 5 chunks
        assert len(rec._raw_buffer) == 5
        # Speech frames: initial 1 + resumed 1 = 2
        assert rec._speech_frames == 2

    def test_vad_exception_treated_as_silence(self) -> None:
        rec = _make_recorder()
        rec._vad_model = MagicMock(side_effect=RuntimeError("vad crash"))

        # Should not raise — exception is caught, treated as silence
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        assert rec.state == _State.WAITING


# ---------------------------------------------------------------------------
# Pre-speech ring buffer
# ---------------------------------------------------------------------------


class TestPreSpeechBuffer:
    def test_pre_speech_chunks_flushed_on_speech_onset(self) -> None:
        rec = _make_recorder()
        silence_val = MagicMock(item=MagicMock(return_value=0.1))
        speech_val = MagicMock(item=MagicMock(return_value=0.9))
        vals = iter([silence_val, silence_val, speech_val])
        rec._vad_model = MagicMock(side_effect=lambda *a, **kw: next(vals))

        # 2 silence chunks go into pre-speech buffer
        rec._audio_callback(_chunk(0.1), 512, None, MagicMock(spec=False))
        rec._audio_callback(_chunk(0.2), 512, None, MagicMock(spec=False))
        assert len(rec._pre_speech_buffer) == 2
        assert len(rec._buffer) == 0

        # Speech chunk flushes pre-speech into main buffer
        rec._audio_callback(_chunk(0.5), 512, None, MagicMock(spec=False))
        assert len(rec._pre_speech_buffer) == 0
        assert len(rec._buffer) == 3  # 2 pre-speech + 1 speech

    def test_pre_speech_buffer_bounded_by_maxlen(self) -> None:
        """Pre-speech buffer should only keep ~500ms worth of chunks."""
        rec = _make_recorder()
        # At 32ms/chunk, 500ms ≈ 15 chunks
        expected_max = max(1, 500 // 32)
        assert rec._pre_speech_buffer.maxlen is None or True  # initial deque

        # After prepare_stream, maxlen is set
        with patch("veery.audio.sd") as mock_sd:
            mock_sd.InputStream = MagicMock()
            rec.prepare_stream()

        assert rec._pre_speech_buffer.maxlen == expected_max


# ---------------------------------------------------------------------------
# Raw buffer (manual stop fallback)
# ---------------------------------------------------------------------------


class TestRawBuffer:
    def test_raw_buffer_always_captures(self) -> None:
        """Raw buffer captures all audio regardless of VAD state."""
        rec = _make_recorder()
        rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.1)))

        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))
        rec._audio_callback(_chunk(), 512, None, MagicMock(spec=False))

        assert len(rec._raw_buffer) == 2
        assert len(rec._buffer) == 0  # no speech, main buffer empty


# ---------------------------------------------------------------------------
# _build_segment
# ---------------------------------------------------------------------------


class TestBuildSegment:
    def test_build_segment_returns_none_when_empty(self) -> None:
        rec = _make_recorder()
        assert rec._build_segment() is None

    def test_build_segment_returns_none_when_too_short(self) -> None:
        vad_cfg = VADConfig(min_speech_duration_sec=1.0)
        rec = _make_recorder(vad_cfg=vad_cfg)
        # Add a tiny chunk (32ms << 1.0s minimum)
        rec._buffer.append(np.zeros(512, dtype=np.float32))
        rec._speech_frames = 3
        assert rec._build_segment() is None

    def test_build_segment_returns_segment_when_valid(self) -> None:
        vad_cfg = VADConfig(min_speech_duration_sec=0.1)
        rec = _make_recorder(vad_cfg=vad_cfg)
        # Add enough audio (16000 samples = 1s)
        rec._buffer.append(np.zeros(16000, dtype=np.float32))
        rec._speech_frames = 3
        seg = rec._build_segment()
        assert seg is not None
        assert seg.duration_sec == 1.0
        assert seg.sample_rate == 16000

    def test_build_segment_strips_trailing_silence(self) -> None:
        vad_cfg = VADConfig(min_speech_duration_sec=0.05)
        audio_cfg = AudioConfig()
        rec = _make_recorder(audio_cfg=audio_cfg, vad_cfg=vad_cfg)

        # Simulate 3 chunks of speech + silence_frames=1
        for _ in range(3):
            rec._buffer.append(np.ones(audio_cfg.chunk_samples, dtype=np.float32))
        rec._speech_frames = 3
        rec._silence_frames = 1

        seg = rec._build_segment()
        assert seg is not None
        # Should be 2 chunks worth (3 minus 1 trailing silence)
        expected_samples = 2 * audio_cfg.chunk_samples
        assert len(seg.audio) == expected_samples

    def test_build_segment_use_raw_fallback(self) -> None:
        vad_cfg = VADConfig(min_speech_duration_sec=0.05)
        rec = _make_recorder(vad_cfg=vad_cfg)

        # Main buffer empty, raw buffer has speech-level audio
        rec._raw_buffer.append(np.full(16000, 0.1, dtype=np.float32))
        rec._state = _State.SPEECH_DETECTED
        assert rec._build_segment(use_raw=False) is None

        seg = rec._build_segment(use_raw=True)
        assert seg is not None
        assert seg.duration_sec == 1.0

    def test_build_segment_use_raw_waiting_low_energy_returns_none(self) -> None:
        """Raw fallback should reject quiet ambient audio in WAITING state."""
        vad_cfg = VADConfig(min_speech_duration_sec=0.05)
        rec = _make_recorder(vad_cfg=vad_cfg)

        # Raw buffer has only low-energy ambient noise.
        rec._raw_buffer.append(np.full(16000, 0.002, dtype=np.float32))
        # _state defaults to WAITING
        assert rec._build_segment(use_raw=True) is None

    def test_build_segment_discards_false_positive_single_speech_frame(self) -> None:
        """Single-frame VAD spikes should not become a valid segment."""
        vad_cfg = VADConfig(min_speech_duration_sec=0.05)
        rec = _make_recorder(vad_cfg=vad_cfg)
        rec._buffer.append(np.ones(16000, dtype=np.float32))
        rec._speech_frames = 1

        assert rec._build_segment() is None


# ---------------------------------------------------------------------------
# prepare_stream / start_recording / stop
# ---------------------------------------------------------------------------


class TestStreamLifecycle:
    def test_prepare_stream_opens_stream(self) -> None:
        rec = _make_recorder()
        with patch("veery.audio.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            rec.prepare_stream()

        assert rec._stream is mock_stream
        mock_stream.start.assert_called_once()

    def test_prepare_stream_noop_if_already_open(self) -> None:
        rec = _make_recorder()
        rec._stream = MagicMock()  # already open
        with patch("veery.audio.sd") as mock_sd:
            rec.prepare_stream()
        mock_sd.InputStream.assert_not_called()

    def test_start_recording_calls_prepare_if_no_stream(self) -> None:
        rec = _make_recorder()
        rec._vad_model = MagicMock()  # pre-loaded
        with patch("veery.audio.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_stream.active = True
            mock_sd.InputStream.return_value = mock_stream
            rec.start_recording()
        assert rec._stream is not None

    def test_stop_recording_closes_stream(self) -> None:
        rec = _make_recorder()
        mock_stream = MagicMock()
        rec._stream = mock_stream

        rec.stop_recording()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert rec._stream is None
        assert rec._done_event.is_set()

    def test_stop_and_flush_uses_raw_fallback(self) -> None:
        vad_cfg = VADConfig(min_speech_duration_sec=0.05)
        rec = _make_recorder(vad_cfg=vad_cfg)
        mock_stream = MagicMock()
        rec._stream = mock_stream
        rec._raw_buffer.append(np.full(16000, 0.1, dtype=np.float32))
        rec._state = _State.SPEECH_DETECTED  # VAD saw speech

        seg = rec.stop_and_flush()

        assert seg is not None
        mock_stream.stop.assert_called_once()

    def test_stop_and_flush_no_speech_returns_none(self) -> None:
        """stop_and_flush returns None when raw audio is just ambient noise."""
        rec = _make_recorder()
        mock_stream = MagicMock()
        rec._stream = mock_stream
        rec._raw_buffer.append(np.zeros(16000, dtype=np.float32))
        # _state remains WAITING (no speech detected)

        seg = rec.stop_and_flush()

        assert seg is None
        mock_stream.stop.assert_called_once()

    def test_stop_and_flush_waiting_with_voice_energy_uses_raw_fallback(self) -> None:
        """Manual stop should recover speech even if VAD never left WAITING."""
        rec = _make_recorder()
        mock_stream = MagicMock()
        rec._stream = mock_stream
        rec._raw_buffer.append(np.full(16000, 0.015, dtype=np.float32))
        rec._state = _State.WAITING

        seg = rec.stop_and_flush()

        assert seg is not None
        assert seg.duration_sec == 1.0
        mock_stream.stop.assert_called_once()

    def test_stop_and_flush_waiting_short_click_rejected(self) -> None:
        """Very short key-click-like bursts should not trigger raw fallback."""
        rec = _make_recorder()
        mock_stream = MagicMock()
        rec._stream = mock_stream
        rec._raw_buffer.append(np.full(512, 0.1, dtype=np.float32))  # ~32ms
        rec._state = _State.WAITING

        seg = rec.stop_and_flush()

        assert seg is None
        mock_stream.stop.assert_called_once()

    def test_is_recording_property(self) -> None:
        rec = _make_recorder()
        assert rec.is_recording is False

        mock_stream = MagicMock()
        mock_stream.active = True
        rec._stream = mock_stream
        assert rec.is_recording is True

    def test_prepare_stream_resets_state(self) -> None:
        rec = _make_recorder()
        rec._state = _State.DONE
        rec._speech_frames = 10
        rec._silence_frames = 5
        rec._done_event.set()

        with patch("veery.audio.sd") as mock_sd:
            mock_sd.InputStream.return_value = MagicMock()
            rec.prepare_stream()

        assert rec.state == _State.WAITING
        assert rec._speech_frames == 0
        assert rec._silence_frames == 0
        assert not rec._done_event.is_set()
