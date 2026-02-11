"""Tests for VeeryApp: state machine, hotkey handling, process_segment, auto-learn."""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from veery.app import _is_repetitive_hallucination
from veery.config import AppConfig

# ---------------------------------------------------------------------------
# _is_repetitive_hallucination
# ---------------------------------------------------------------------------


class TestIsRepetitiveHallucination:
    def test_empty_string(self) -> None:
        assert _is_repetitive_hallucination("") is False

    def test_short_text_not_flagged(self) -> None:
        """Fewer than 6 words should never be flagged."""
        assert _is_repetitive_hallucination("Why Why Why Why Why") is False

    def test_boundary_six_words_below_threshold(self) -> None:
        """6 words where no single word exceeds 80%."""
        assert _is_repetitive_hallucination("a b c d e f") is False

    def test_boundary_six_words_at_threshold(self) -> None:
        """6 words, 5 repetitions = 83% > 80% → hallucination."""
        assert _is_repetitive_hallucination("Why Why Why Why Why ok") is True

    def test_classic_hallucination(self) -> None:
        assert _is_repetitive_hallucination("Why Why Why Why Why Why Why Why") is True

    def test_case_insensitive(self) -> None:
        """Mixed casing should still be detected."""
        assert _is_repetitive_hallucination("Thank thank THANK Thank thank Thank") is True

    def test_legitimate_repetitive_speech(self) -> None:
        """Legitimate text with some repetition but under 80%."""
        assert _is_repetitive_hallucination("go go go team go let us win") is False

    def test_normal_sentence(self) -> None:
        assert _is_repetitive_hallucination(
            "The quick brown fox jumps over the lazy dog"
        ) is False

    def test_mixed_language_not_flagged(self) -> None:
        """Bilingual text should not be falsely flagged."""
        assert _is_repetitive_hallucination(
            "我们 today 讨论 the project plan together"
        ) is False

    def test_hallucination_discards_in_process_segment(self, app) -> None:
        """Integration: hallucination is discarded and shows correct warning."""
        from veery.app import State

        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "Why Why Why Why Why Why Why"
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        app._overlay.show_warning.assert_called_with("Filtered repetitive audio")
        assert app._state == State.IDLE
        assert app._session_count == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a VeeryApp with all heavy dependencies mocked.

    rumps is left real (it can construct MenuItems without NSApplication).
    Only ML models, audio, overlays, hotkey, and sounds are mocked.
    """
    patches = [
        patch("veery.app.AudioRecorder"),
        patch("veery.app.JargonCorrector"),
        patch("veery.app.TextCorrector"),
        patch("veery.app.CorrectionLearner"),
        patch("veery.app.OverlayIndicator"),
        patch("veery.app.PermissionGuideOverlay"),
        patch("veery.app.DownloadProgressOverlay"),
        patch("veery.app.check_permissions_granted", return_value=True),
        patch("veery.app.sounds"),
        patch("veery.app.create_stt"),
        patch("veery.app._is_model_cached", return_value=True),
        patch("veery.app._is_sensevoice_cached", return_value=True),
        patch("veery.app.ensure_model_downloaded"),
        patch("veery.app.ensure_sensevoice_downloaded"),
        patch("veery.app.paste_to_active_app"),
        # Prevent real hotkey listener from starting
        patch("veery.app.VeeryApp._start_hotkey_listener"),
        # Prevent background thread for model loading — we call it synchronously
        patch("veery.app.threading.Thread"),
    ]

    for p in patches:
        p.start()

    from veery.app import VeeryApp

    vf_app = VeeryApp(AppConfig())
    # Simulate model loading complete
    vf_app._ready.set()
    # Give it a mock STT
    vf_app._stt = MagicMock()

    yield vf_app

    for p in patches:
        p.stop()


@dataclass
class FakeSegment:
    audio: np.ndarray
    sample_rate: int
    duration_sec: float = 1.0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_initial_state_is_idle(self, app) -> None:
        from veery.app import State
        assert app._state == State.IDLE

    def test_set_state_recording(self, app) -> None:
        from veery.app import State
        app._set_state(State.RECORDING)
        assert app._state == State.RECORDING
        app._overlay.show_recording.assert_called_once()

    def test_set_state_processing(self, app) -> None:
        from veery.app import State
        app._set_state(State.PROCESSING)
        assert app._state == State.PROCESSING
        app._overlay.show_processing.assert_called_once()

    def test_set_state_idle_hides_overlay(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._set_state(State.IDLE)
        assert app._state == State.IDLE
        app._overlay.hide.assert_called()

    def test_set_state_idle_skip_overlay(self, app) -> None:
        from veery.app import State
        app._overlay.hide.reset_mock()
        app._set_state(State.IDLE, skip_overlay=True)
        app._overlay.hide.assert_not_called()


# ---------------------------------------------------------------------------
# _begin_recording
# ---------------------------------------------------------------------------


class TestBeginRecording:
    def test_begin_recording_from_idle(self, app) -> None:
        from veery.app import State

        # Unpatch threading.Thread so _begin_recording can spawn its thread
        with patch("veery.app.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            app._begin_recording()

        assert app._state == State.RECORDING

    def test_begin_recording_noop_when_not_idle(self, app) -> None:
        from veery.app import State
        app._state = State.RECORDING
        app._recorder.prepare_stream.reset_mock()

        with patch("veery.app.threading.Thread"):
            app._begin_recording()

        app._recorder.prepare_stream.assert_not_called()

    def test_begin_recording_prepare_stream_failure(self, app) -> None:
        from veery.app import State
        app._recorder.prepare_stream.side_effect = RuntimeError("mic error")

        with patch("veery.app.threading.Thread"):
            app._begin_recording()

        assert app._state == State.IDLE
        assert app._recording_started.is_set()


# ---------------------------------------------------------------------------
# _on_key_down / _on_key_up (hold mode)
# ---------------------------------------------------------------------------


class TestHoldMode:
    def test_key_down_ignored_before_ready(self, app) -> None:
        from veery.app import State
        app._ready.clear()
        app._on_key_down()
        assert app._state == State.IDLE

    def test_key_up_ignored_in_toggle_mode(self, app) -> None:
        from veery.app import State
        app._recording_mode = "toggle"
        app._state = State.RECORDING
        app._on_key_up()
        assert app._state == State.RECORDING

    def test_key_up_stops_recording_in_hold_mode(self, app) -> None:
        app._state = app._state.__class__("recording")  # State.RECORDING
        app._recording_started.set()
        app._recorder.stop_and_flush.return_value = None

        app._on_key_up()

        app._recorder.stop_and_flush.assert_called_once()

    def test_key_up_noop_when_idle(self, app) -> None:
        from veery.app import State
        app._recording_mode = "hold"
        app._state = State.IDLE
        app._on_key_up()
        app._recorder.stop_and_flush.assert_not_called()


# ---------------------------------------------------------------------------
# _on_toggle_key
# ---------------------------------------------------------------------------


class TestToggleMode:
    def test_toggle_starts_recording_from_idle(self, app) -> None:
        from veery.app import State
        app._recording_mode = "toggle"

        with patch("veery.app.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            app._on_key_down()

        assert app._state == State.RECORDING

    def test_toggle_stops_recording(self, app) -> None:
        from veery.app import State
        app._recording_mode = "toggle"
        app._state = State.RECORDING
        app._recording_started.set()
        app._recorder.stop_and_flush.return_value = None

        app._on_toggle_key()

        app._recorder.stop_and_flush.assert_called_once()

    def test_toggle_noop_during_processing(self, app) -> None:
        from veery.app import State
        app._recording_mode = "toggle"
        app._state = State.PROCESSING
        app._on_toggle_key()
        app._recorder.prepare_stream.assert_not_called()
        app._recorder.stop_and_flush.assert_not_called()


# ---------------------------------------------------------------------------
# _stop_recording
# ---------------------------------------------------------------------------


class TestStopRecording:
    def test_stop_recording_no_speech(self, app) -> None:
        from veery.app import State
        app._state = State.RECORDING
        app._recorder.stop_and_flush.return_value = None

        app._stop_recording()

        app._overlay.show_warning.assert_called_with("No speech detected")
        assert app._state == State.IDLE

    def test_stop_recording_with_speech_transitions_to_processing(self, app) -> None:
        from veery.app import State
        app._state = State.RECORDING
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._recorder.stop_and_flush.return_value = seg

        with patch("veery.app.threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            app._stop_recording()

        assert app._state == State.PROCESSING
        mock_instance.start.assert_called_once()

    def test_stop_recording_no_recorder(self, app) -> None:
        from veery.app import State
        app._recorder = None
        app._state = State.RECORDING
        app._stop_recording()
        assert app._state == State.IDLE


# ---------------------------------------------------------------------------
# _process_segment
# ---------------------------------------------------------------------------


class TestProcessSegment:
    def test_process_segment_success(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._stt.transcribe.return_value = "hello world"
        mock_result = MagicMock()
        mock_result.final = "hello world"
        app._corrector.correct.return_value = mock_result

        app._process_segment(seg)

        assert app._state == State.IDLE
        assert app._session_count == 1
        assert app._last_pasted_text == "hello world"
        app._overlay.show_success.assert_called_once()

    def test_process_segment_stt_none(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt = None
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_empty_transcription(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.return_value = ""
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        app._overlay.show_warning.assert_called_with("No transcription")
        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_short_low_energy_skips_stt(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        seg = FakeSegment(audio=np.zeros(12000, dtype=np.float32), sample_rate=16000, duration_sec=0.75)

        app._process_segment(seg)

        app._stt.transcribe.assert_not_called()
        app._overlay.show_warning.assert_called_with("No speech detected")
        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_stt_exception(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.side_effect = RuntimeError("STT crash")
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_no_corrector(self, app) -> None:
        """When corrector is None, raw text is pasted directly."""
        from veery.app import State
        app._state = State.PROCESSING
        app._corrector = None
        app._stt.transcribe.return_value = "raw text"
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        assert app._session_count == 1
        assert app._last_pasted_text == "raw text"

    def test_process_segment_local_stt_reference(self, app) -> None:
        """Verify _process_segment uses a local reference to _stt."""
        from veery.app import State
        app._state = State.PROCESSING

        original_stt = MagicMock()
        original_stt.transcribe.return_value = "hello"
        app._stt = original_stt

        mock_result = MagicMock()
        mock_result.final = "hello"
        app._corrector.correct.return_value = mock_result

        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._process_segment(seg)

        original_stt.transcribe.assert_called_once()

    def test_session_count_updates_detail(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "hello"
        mock_result = MagicMock()
        mock_result.final = "hello"
        app._corrector.correct.return_value = mock_result
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg)

        assert app._session_count == 1
        assert "1 dictation" in app._detail_item.title


# ---------------------------------------------------------------------------
# _try_auto_learn
# ---------------------------------------------------------------------------


class TestTryAutoLearn:
    def test_auto_learn_skipped_no_learner(self, app) -> None:
        app._learner = None
        app._last_pasted_text = "old text"
        app._last_pasted_time = time.monotonic()
        app._try_auto_learn("new text")  # should not crash

    def test_auto_learn_skipped_no_previous(self, app) -> None:
        app._last_pasted_text = None
        app._try_auto_learn("new text")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_skipped_outside_window(self, app) -> None:
        app._last_pasted_text = "old text"
        app._last_pasted_time = time.monotonic() - 60
        app._try_auto_learn("new text")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_skipped_too_different(self, app) -> None:
        app._last_pasted_text = "completely unrelated text"
        app._last_pasted_time = time.monotonic()
        with patch("rapidfuzz.fuzz.ratio", return_value=10):
            app._try_auto_learn("xyz abc 123")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_skipped_too_similar(self, app) -> None:
        app._last_pasted_text = "hello world"
        app._last_pasted_time = time.monotonic()
        with patch("rapidfuzz.fuzz.ratio", return_value=98):
            app._try_auto_learn("hello world")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_logs_correction(self, app) -> None:
        app._last_pasted_text = "sharp ratio"
        app._last_pasted_time = time.monotonic()
        app._learner.log_correction.return_value = None

        with patch("rapidfuzz.fuzz.ratio", return_value=70):
            app._try_auto_learn("Sharpe ratio")

        app._learner.log_correction.assert_called_once_with("sharp ratio", "Sharpe ratio")


# ---------------------------------------------------------------------------
# Menu callbacks
# ---------------------------------------------------------------------------


class TestMenuCallbacks:
    def test_toggle_mode_switches(self, app) -> None:
        assert app._recording_mode == "hold"
        app._on_toggle_mode(None)
        assert app._recording_mode == "toggle"
        app._on_toggle_mode(None)
        assert app._recording_mode == "hold"

    def test_hotkey_hint_hold(self, app) -> None:
        app._recording_mode = "hold"
        assert "Hold" in app._hotkey_hint()

    def test_hotkey_hint_toggle(self, app) -> None:
        app._recording_mode = "toggle"
        assert "start/stop" in app._hotkey_hint()

    def test_mode_label_hold(self, app) -> None:
        app._recording_mode = "hold"
        assert "Toggle" in app._mode_label()

    def test_mode_label_toggle(self, app) -> None:
        app._recording_mode = "toggle"
        assert "Hold" in app._mode_label()


# ---------------------------------------------------------------------------
# _on_quit
# ---------------------------------------------------------------------------


class TestOnQuit:
    def test_quit_stops_listener_and_recorder(self, app) -> None:
        mock_listener = MagicMock()
        app._hotkey_listener = mock_listener
        app._recorder.is_recording = True

        with patch("veery.app.rumps"):
            app._on_quit(None)

        mock_listener.stop.assert_called_once()
        app._recorder.stop_recording.assert_called_once()

    def test_quit_no_recorder(self, app) -> None:
        app._recorder = None
        app._hotkey_listener = None
        with patch("veery.app.rumps"):
            app._on_quit(None)  # should not crash


# ---------------------------------------------------------------------------
# STT backend switching
# ---------------------------------------------------------------------------


class TestSTTBackendSwitch:
    def test_select_same_backend_noop(self, app) -> None:
        app._stt_backend = "whisper"
        app._stt = MagicMock()  # not None
        app._on_select_stt_backend("whisper", None)
        # Should return immediately, no thread spawned

    def test_select_different_backend(self, app) -> None:
        app._stt_backend = "sensevoice"

        with patch("veery.app.threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            app._on_select_stt_backend("whisper", None)

        # Backend only updates after the switch thread completes (no race)
        assert app._stt_backend == "sensevoice"
        mock_instance.start.assert_called_once()

    def test_switch_cancels_whisper_download(self, app) -> None:
        app._whisper_loading = True
        app._whisper_download_cancelled = False

        with patch("veery.app.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            app._on_select_stt_backend("sensevoice", None)

        assert app._whisper_download_cancelled is True

    def test_select_whisper_while_downloading_lets_background_finish(self, app) -> None:
        app._whisper_loading = True
        app._whisper_download_cancelled = True  # was previously cancelled
        app._stt_backend = "sensevoice"

        with patch("veery.app.threading.Thread") as mock_thread:
            app._on_select_stt_backend("whisper", None)

        # Should NOT spawn a new thread — let the background download auto-switch
        mock_thread.assert_not_called()
        assert app._whisper_download_cancelled is False
