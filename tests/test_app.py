"""Tests for VeeryApp: state machine, hotkey handling, process_segment, auto-learn."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from veery.app import VeeryApp, _is_repetitive_hallucination
from veery.audio import StopReason
from veery.config import AppConfig, STTConfig
from veery.jargon import JargonUsageTracker

_ORIGINAL_START_HOTKEY_LISTENER = VeeryApp._start_hotkey_listener

# ---------------------------------------------------------------------------
# _is_repetitive_hallucination
# ---------------------------------------------------------------------------


class TestIsRepetitiveHallucination:
    def test_empty_string(self) -> None:
        assert _is_repetitive_hallucination("") is False

    def test_short_text_not_flagged(self) -> None:
        """Fewer than 8 words should never be flagged."""
        assert _is_repetitive_hallucination("Why Why Why Why Why Why Why") is False

    def test_six_repeated_words_pass_through(self) -> None:
        """Deliberate six-word repetition is below the new length gate."""
        assert _is_repetitive_hallucination("yes yes yes yes yes yes") is False

    def test_boundary_eight_words_above_threshold(self) -> None:
        """8 words with 7 repetitions = 87.5% > 85%."""
        assert _is_repetitive_hallucination("Why Why Why Why Why Why Why ok") is True

    def test_classic_hallucination(self) -> None:
        assert _is_repetitive_hallucination("Why Why Why Why Why Why Why Why") is True

    def test_case_insensitive(self) -> None:
        """Mixed casing should still be detected."""
        assert _is_repetitive_hallucination(
            "Thank thank THANK Thank thank Thank THANK thank"
        ) is True

    def test_legitimate_repetitive_speech(self) -> None:
        """Legitimate text with some repetition but under 85%."""
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
        app._stt.transcribe.return_value = "Why Why Why Why Why Why Why Why"
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._overlay.hide.reset_mock()

        app._process_segment(seg, generation=1)

        app._overlay.show_warning.assert_called_with("Filtered repetitive audio")
        app._overlay.hide.assert_not_called()
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
        patch("veery.app.JargonUsageTracker"),
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
        patch("veery.app._check_accessibility", return_value=True),
        # Prevent real hotkey listener from starting
        patch("veery.app.VeeryApp._start_hotkey_listener"),
        # Prevent background thread for model loading — we call it synchronously
        patch("veery.app.threading.Thread"),
    ]

    for p in patches:
        p.start()

    vf_app = VeeryApp(AppConfig())
    # Simulate model loading complete
    vf_app._ready.set()
    # Give it a mock STT
    vf_app._stt = MagicMock()
    # Direct _process_segment calls below simulate the first live worker.
    vf_app._processing_generation = 1
    # Keep prompt-ranking tests deterministic while avoiding real stats I/O.
    vf_app._usage_tracker.rank.side_effect = lambda terms: list(terms)

    yield vf_app

    for p in patches:
        p.stop()


@dataclass
class FakeSegment:
    audio: np.ndarray
    sample_rate: int
    duration_sec: float = 1.0


@dataclass
class FakeCharKey:
    char: str


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

    @pytest.mark.parametrize("state_name", ["IDLE", "RECORDING"])
    def test_leave_processing_state_rejects_non_processing_state(
        self, app, state_name: str
    ) -> None:
        from veery.app import State

        original_state = State[state_name]
        app._state = original_state

        assert app._leave_processing_state(skip_overlay=True) is False
        assert app._state is original_state

    def test_leave_processing_state_changes_processing_to_idle(self, app) -> None:
        from veery.app import State

        app._state = State.PROCESSING

        assert app._leave_processing_state(skip_overlay=True) is True
        assert app._state is State.IDLE


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
        app._recorder.prepare_stream.assert_called_once_with(manual_mode=True)

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

        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)

    def test_key_up_does_not_wait_for_background_start(self, app) -> None:
        app._state = app._state.__class__("recording")  # State.RECORDING
        app._recording_started.wait = MagicMock(return_value=False)
        app._recorder.stop_and_flush.return_value = None

        app._on_key_up()

        app._recording_started.wait.assert_not_called()
        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)

    def test_key_up_noop_when_idle(self, app) -> None:
        from veery.app import State
        app._recording_mode = "hold"
        app._state = State.IDLE
        app._on_key_up()
        app._recorder.stop_and_flush.assert_not_called()


class TestHotkeyListener:
    def test_press_callback_contains_handler_exceptions(self, app) -> None:
        target_key = object()
        listener = MagicMock()
        listener.is_alive.return_value = True
        listener_factory = MagicMock(return_value=listener)
        keyboard_module = SimpleNamespace(
            Key=SimpleNamespace(cmd_r=target_key),
            Listener=listener_factory,
        )
        pynput_module = SimpleNamespace(keyboard=keyboard_module)
        app._on_key_down = MagicMock(side_effect=RuntimeError("key handler failed"))

        with (
            patch.dict(
                "sys.modules",
                {"pynput": pynput_module, "pynput.keyboard": keyboard_module},
            ),
            patch("veery.app.time.sleep"),
        ):
            _ORIGINAL_START_HOTKEY_LISTENER(app)

        on_press = listener_factory.call_args.kwargs["on_press"]
        on_press(target_key)

        app._on_key_down.assert_called_once_with()


# ---------------------------------------------------------------------------
# Permission flow
# ---------------------------------------------------------------------------


class TestPermissionFlow:
    def test_permissions_granted_restarts_dead_hotkey_listener(self, app) -> None:
        dead_listener = MagicMock()
        dead_listener.is_alive.return_value = False
        app._hotkey_listener = dead_listener

        with patch.object(app, "_start_hotkey_listener") as mock_start:
            app._on_permissions_granted()

        mock_start.assert_called_once()


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

        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)

    def test_toggle_stop_does_not_wait_for_background_start(self, app) -> None:
        from veery.app import State
        app._recording_mode = "toggle"
        app._state = State.RECORDING
        app._recording_started.wait = MagicMock(return_value=False)
        app._recorder.stop_and_flush.return_value = None

        app._on_toggle_key()

        app._recording_started.wait.assert_not_called()
        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)

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


class TestStartRecording:
    def test_start_recording_uses_manual_mode(self, app) -> None:
        app._start_recording()

        app._recorder.start_recording.assert_called_once_with(manual_mode=True, open_stream_if_needed=False)

    def test_start_recording_spawns_manual_cap_watcher_when_configured(self, app) -> None:
        from veery.app import State

        app._state = State.RECORDING
        app._config = replace(
            app._config,
            audio=replace(app._config.audio, manual_max_duration_sec=60.0),
        )

        with patch("veery.app.threading.Thread") as mock_thread:
            watcher = MagicMock()
            mock_thread.return_value = watcher
            app._start_recording()

        app._recorder.start_recording.assert_called_once_with(manual_mode=True, open_stream_if_needed=False)
        watcher.start.assert_called_once()


class TestManualCapWatcher:
    def test_watch_manual_stop_ignores_user_stop(self, app) -> None:
        app._recorder.wait_for_manual_stop.return_value = StopReason.USER_STOP

        with patch.object(app, "_stop_recording") as mock_stop:
            app._watch_manual_stop()

        mock_stop.assert_not_called()

    def test_watch_manual_stop_handles_manual_cap(self, app) -> None:
        from veery.app import State

        app._state = State.RECORDING
        app._recorder.wait_for_manual_stop.return_value = StopReason.MANUAL_CAP_REACHED

        with patch.object(app, "_stop_recording") as mock_stop:
            app._watch_manual_stop()

        mock_stop.assert_called_once_with(reason=StopReason.MANUAL_CAP_REACHED)


class TestStopRecording:
    def test_stop_recording_no_speech(self, app) -> None:
        from veery.app import State
        app._state = State.RECORDING
        app._recorder.stop_and_flush.return_value = None

        app._stop_recording()

        app._overlay.show_warning.assert_called_with("No speech detected")
        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)
        assert app._state == State.IDLE

    def test_stop_recording_with_speech_transitions_to_processing(self, app) -> None:
        from veery.app import State
        app._state = State.RECORDING
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._recorder.stop_and_flush.return_value = seg

        with patch("veery.app.threading.Thread") as mock_thread:
            worker = MagicMock()
            watchdog = MagicMock()
            mock_thread.side_effect = [worker, watchdog]
            app._stop_recording()

        assert app._state == State.PROCESSING
        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.USER_STOP)
        assert mock_thread.call_count == 2
        worker.start.assert_called_once()
        watchdog.start.assert_called_once()
        assert mock_thread.call_args_list[1].kwargs["target"] == app._watch_processing
        assert mock_thread.call_args_list[1].kwargs["args"][0] is worker

    def test_stop_recording_manual_cap_shows_message_and_processes(self, app) -> None:
        from veery.app import State

        app._state = State.RECORDING
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._recorder.stop_and_flush.return_value = seg

        with (
            patch("veery.app.threading.Thread") as mock_thread,
            patch("veery.app.rumps.notification") as mock_notification,
        ):
            worker = MagicMock()
            watchdog = MagicMock()
            mock_thread.side_effect = [worker, watchdog]
            app._stop_recording(reason=StopReason.MANUAL_CAP_REACHED)

        assert app._state == State.PROCESSING
        app._recorder.stop_and_flush.assert_called_once_with(reason=StopReason.MANUAL_CAP_REACHED)
        mock_notification.assert_called_once_with(
            "Veery",
            "Recording stopped",
            "Max dictation length reached",
        )
        assert mock_thread.call_count == 2
        worker.start.assert_called_once()
        watchdog.start.assert_called_once()

    def test_stop_recording_flush_failure_returns_to_idle(self, app) -> None:
        from veery.app import State

        app._state = State.RECORDING
        app._recorder.stop_and_flush.side_effect = RuntimeError("device disappeared")
        app._overlay.hide.reset_mock()

        app._stop_recording()

        assert app._state == State.IDLE
        assert app._recording_finalizing is False
        app._overlay.show_warning.assert_called_once_with("Recording failed")
        app._overlay.hide.assert_not_called()

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

        with patch("veery.app._check_accessibility", return_value=True) as check_accessibility:
            app._process_segment(seg, generation=1)

        assert app._state == State.IDLE
        assert app._session_count == 1
        assert app._last_pasted_text == "hello world"
        app._overlay.show_success.assert_called_once()
        check_accessibility.assert_called_once_with()

    def test_process_segment_stt_none(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt = None
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg, generation=1)

        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_empty_transcription(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.return_value = ""
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        app._overlay.hide.reset_mock()

        app._process_segment(seg, generation=1)

        app._overlay.show_warning.assert_called_with("No speech detected")
        app._overlay.hide.assert_not_called()
        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_vad_confirmed_quiet_speech_reaches_stt(self, app) -> None:
        import veery.app as app_module
        from veery.app import State

        app._state = State.PROCESSING
        app._recorder.has_speech.return_value = True
        app._stt.transcribe.return_value = "quiet speech"
        result = MagicMock(final="quiet speech")
        app._corrector.correct.return_value = result
        seg = FakeSegment(audio=np.zeros(12000, dtype=np.float32), sample_rate=16000, duration_sec=0.75)

        app_module.paste_to_active_app.reset_mock()
        with patch("veery.app._check_accessibility", return_value=True):
            app._process_segment(seg, generation=1)

        app._stt.transcribe.assert_called_once()
        app_module.paste_to_active_app.assert_called_once_with("quiet speech", app._config.output)
        assert app._state == State.IDLE
        assert app._session_count == 1

    @pytest.mark.parametrize("vad_unavailable", ["no_recorder", "vad_error"])
    def test_process_segment_short_low_energy_guard_when_vad_unavailable(
        self, app, vad_unavailable: str
    ) -> None:
        from veery.app import State

        app._state = State.PROCESSING
        if vad_unavailable == "no_recorder":
            app._recorder = None
        else:
            app._recorder.has_speech.side_effect = RuntimeError("VAD unavailable")
        seg = FakeSegment(
            audio=np.zeros(12000, dtype=np.float32),
            sample_rate=16000,
            duration_sec=0.75,
        )

        app._process_segment(seg, generation=1)

        app._stt.transcribe.assert_not_called()
        app._overlay.show_warning.assert_called_once_with("No speech detected")
        assert app._state == State.IDLE

    def test_process_segment_vad_gate_rejects_no_speech(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._recorder.has_speech.return_value = False
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._overlay.hide.reset_mock()
        app._process_segment(seg, generation=1)

        app._stt.transcribe.assert_not_called()
        app._overlay.show_warning.assert_called_with("No speech detected")
        app._overlay.hide.assert_not_called()
        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_stt_exception(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.side_effect = RuntimeError("STT crash")
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg, generation=1)

        assert app._state == State.IDLE
        assert app._session_count == 0

    def test_process_segment_stt_error_reports_transcription_failure(self, app) -> None:
        from veery.app import State
        from veery.stt import STTError

        app._state = State.PROCESSING
        app._recorder.has_speech.return_value = True
        app._stt.transcribe.side_effect = STTError("backend unavailable")
        app._overlay.show_warning.reset_mock()
        seg = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg, generation=1)

        warning = app._overlay.show_warning.call_args.args[0]
        assert "Transcription failed" in warning
        assert "No speech detected" not in warning
        assert app._state is State.IDLE

    def test_process_segment_cancelled_during_correction_does_not_paste(self, app) -> None:
        from veery.app import State

        app._state = State.PROCESSING
        app._recorder.has_speech.return_value = True
        app._stt.transcribe.return_value = "hello"
        seg = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)

        def cancel_generation(_raw_text: str) -> MagicMock:
            with app._processing_lock:
                app._cancelled_processing_generation = 1
            return MagicMock(final="hello")

        app._corrector.correct.side_effect = cancel_generation

        with (
            patch("veery.app._check_accessibility", return_value=True),
            patch("veery.app.paste_to_active_app") as paste,
        ):
            app._process_segment(seg, generation=1)

        paste.assert_not_called()

    def test_process_segment_no_corrector(self, app) -> None:
        """When corrector is None, raw text is pasted directly."""
        from veery.app import State
        app._state = State.PROCESSING
        app._corrector = None
        app._stt.transcribe.return_value = "raw text"
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        with patch("veery.app._check_accessibility", return_value=True):
            app._process_segment(seg, generation=1)

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
        with patch("veery.app._check_accessibility", return_value=True):
            app._process_segment(seg, generation=1)

        original_stt.transcribe.assert_called_once()

    def test_session_count_updates_detail(self, app) -> None:
        from veery.app import State
        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "hello"
        mock_result = MagicMock()
        mock_result.final = "hello"
        app._corrector.correct.return_value = mock_result
        seg = FakeSegment(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000)

        with patch("veery.app._check_accessibility", return_value=True):
            app._process_segment(seg, generation=1)

        assert app._session_count == 1
        assert "1 dictation" in app._detail_item.title

    def test_process_segment_empty_corrected_text_keeps_warning_visible(self, app) -> None:
        from veery.app import State

        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "um"
        app._corrector.correct.return_value = MagicMock(final="")
        app._overlay.hide.reset_mock()
        seg = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)

        app._process_segment(seg, generation=1)

        app._overlay.show_warning.assert_called_once_with("Nothing left after cleanup")
        app._overlay.hide.assert_not_called()

    def test_process_segment_missing_accessibility_does_not_paste(self, app) -> None:
        from veery.app import State

        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "hello"
        app._corrector.correct.return_value = MagicMock(final="hello")
        seg = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)

        with (
            patch("veery.app._check_accessibility", return_value=False),
            patch("veery.app.paste_to_active_app") as paste,
        ):
            app._process_segment(seg, generation=1)

        paste.assert_not_called()
        app._overlay.show_warning.assert_called_once_with("No Accessibility permission")
        assert app._session_count == 0

    def test_paste_precedes_non_fatal_learning_and_usage_tracking(self, app) -> None:
        from veery.app import State

        app._state = State.PROCESSING
        app._stt.transcribe.return_value = "hello"
        app._corrector.correct.return_value = MagicMock(final="hello")
        seg = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)
        events: list[str] = []

        def fail_auto_learn(_text: str) -> None:
            events.append("auto_learn")
            raise RuntimeError("learning failed")

        def fail_usage_tracking(_text: str) -> None:
            events.append("usage")
            raise RuntimeError("tracking failed")

        with (
            patch("veery.app._check_accessibility", return_value=True),
            patch("veery.app.paste_to_active_app", side_effect=lambda *_: events.append("paste")),
            patch.object(app, "_try_auto_learn", side_effect=fail_auto_learn),
            patch.object(app, "_record_jargon_usage", side_effect=fail_usage_tracking),
        ):
            app._process_segment(seg, generation=1)

        assert events == ["paste", "auto_learn", "usage"]
        assert app._session_count == 1
        app._overlay.show_success.assert_called_once_with()


class TestProcessingWatchdog:
    def _start_blocked_worker(self, app) -> tuple[Thread, Event]:
        import veery.app as app_module
        from veery.app import State

        release = Event()
        app._config = replace(
            app._config,
            stt=replace(app._config.stt, processing_timeout_sec=0.05),
        )
        app._state = State.PROCESSING
        app._processing_generation = 1

        def blocked_transcribe(*_args) -> str:
            entered.set()
            release.wait(timeout=1)
            return "late text"

        entered = Event()
        app._stt.transcribe.side_effect = blocked_transcribe
        app._corrector.correct.return_value = MagicMock(final="late text")
        app_module.paste_to_active_app.reset_mock()
        segment = FakeSegment(audio=np.ones(16000, dtype=np.float32), sample_rate=16000)
        worker = Thread(target=app._process_segment, args=(segment, 1), daemon=True)
        worker.start()
        assert entered.wait(timeout=1)
        return worker, release

    def test_watchdog_resets_processing_state_to_idle(self, app) -> None:
        from veery.app import State

        worker, release = self._start_blocked_worker(app)
        try:
            app._watch_processing(worker, generation=1)
            assert app._state == State.IDLE
            assert app._is_processing_cancelled(1) is True
            app._overlay.show_warning.assert_called_once_with("Transcription timed out")
        finally:
            release.set()
            worker.join(timeout=1)

    def test_late_worker_result_is_discarded_without_paste(self, app) -> None:
        import veery.app as app_module

        worker, release = self._start_blocked_worker(app)
        try:
            app._watch_processing(worker, generation=1)
            release.set()
            worker.join(timeout=1)
            assert worker.is_alive() is False
            app_module.paste_to_active_app.assert_not_called()
            assert app._session_count == 0
        finally:
            release.set()
            worker.join(timeout=1)

    def test_late_worker_does_not_stomp_newer_processing_session(self, app) -> None:
        import veery.app as app_module
        from veery.app import State

        worker, release = self._start_blocked_worker(app)
        try:
            app._watch_processing(worker, generation=1)
            with app._processing_lock:
                app._processing_generation = 2
                app._state = State.PROCESSING

            release.set()
            worker.join(timeout=1)

            assert worker.is_alive() is False
            assert app._state is State.PROCESSING
            app_module.paste_to_active_app.assert_not_called()
        finally:
            release.set()
            worker.join(timeout=1)

    def test_completed_worker_is_not_cancelled_or_reported_as_timed_out(self, app) -> None:
        worker = Thread(target=lambda: None, daemon=True)
        worker.start()
        worker.join(timeout=1)
        app._overlay.show_warning.reset_mock()

        with patch.object(app, "_notify") as notify:
            app._watch_processing(worker, generation=1)

        assert app._is_processing_cancelled(1) is False
        notify.assert_not_called()
        app._overlay.show_warning.assert_not_called()


class TestJargonUsageRecording:
    @staticmethod
    def _use_real_tracker(app, tmp_path) -> JargonUsageTracker:
        tracker = JargonUsageTracker(tmp_path / "usage_stats.yaml")
        app._usage_tracker = tracker
        app._corrector.jargon.dictionary.canonical_terms = [
            "API",
            "DuckDB",
            "夏普比率",
        ]
        return tracker

    def test_ascii_term_is_recorded_on_word_boundary(self, app, tmp_path) -> None:
        tracker = self._use_real_tracker(app, tmp_path)

        app._record_jargon_usage("the API call")

        assert tracker._stats["API"]["count"] == 1
        assert "DuckDB" not in tracker._stats

    def test_ascii_term_is_not_recorded_inside_another_word(self, app, tmp_path) -> None:
        tracker = self._use_real_tracker(app, tmp_path)

        app._record_jargon_usage("capitalized calls")

        assert tracker._stats == {}

    def test_cjk_term_is_recorded_by_substring(self, app, tmp_path) -> None:
        tracker = self._use_real_tracker(app, tmp_path)

        app._record_jargon_usage("用夏普比率算一下")

        assert tracker._stats["夏普比率"]["count"] == 1

    def test_fifth_record_refreshes_runtime_hints(self, app, tmp_path) -> None:
        self._use_real_tracker(app, tmp_path)

        with patch.object(app, "_apply_stt_runtime_hints") as apply_hints:
            for _ in range(4):
                app._record_jargon_usage("the API call")
            apply_hints.assert_not_called()

            app._record_jargon_usage("the API call")

        apply_hints.assert_called_once_with(app._stt)


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

    def test_auto_learn_skipped_when_same_text(self, app) -> None:
        app._last_pasted_text = "hello world"
        app._last_pasted_time = time.monotonic()
        with patch("rapidfuzz.fuzz.ratio", return_value=98):
            app._try_auto_learn("hello world")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_high_similarity_still_logs_when_different(self, app) -> None:
        app._last_pasted_text = "the sharp ratio is up today"
        app._last_pasted_time = time.monotonic()
        app._learner.log_correction.return_value = None

        # Minor edits can still be valid corrections and often score >95.
        with patch("rapidfuzz.fuzz.ratio", return_value=99):
            app._try_auto_learn("the Sharpe ratio is up today")

        app._learner.log_correction.assert_called_once_with(
            "the sharp ratio is up today",
            "the Sharpe ratio is up today",
        )

    def test_auto_learn_logs_correction(self, app) -> None:
        app._last_pasted_text = "sharp ratio"
        app._last_pasted_time = time.monotonic()
        app._learner.log_correction.return_value = None

        with patch("rapidfuzz.fuzz.ratio", return_value=70):
            app._try_auto_learn("Sharpe ratio")

        app._learner.log_correction.assert_called_once_with("sharp ratio", "Sharpe ratio")

    def test_auto_learn_reloads_corrector_on_promotion(self, app) -> None:
        import veery.app as app_module

        app._last_pasted_text = "sharp ratio"
        app._last_pasted_time = time.monotonic()
        app._learner.log_correction.return_value = "Sharpe ratio"
        app_module.JargonCorrector.reset_mock()
        app_module.TextCorrector.reset_mock()

        with patch("rapidfuzz.fuzz.ratio", return_value=70):
            app._try_auto_learn("Sharpe ratio")

        app_module.JargonCorrector.assert_called_once_with(app._config.jargon)
        app_module.TextCorrector.assert_called_once()


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


class TestNotifications:
    def test_notify_swallows_rumps_runtime_error(self, app) -> None:
        with patch(
            "veery.app.rumps.notification",
            side_effect=RuntimeError("not running from an app bundle"),
        ):
            app._notify("Veery", "Test", "message")


# ---------------------------------------------------------------------------
# Manual edit learning
# ---------------------------------------------------------------------------


class TestManualEditLearning:
    def test_cmd_arrow_delete_type_workflow_logs_correction(self, app) -> None:
        app._learner.log_correction.return_value = None
        app._begin_manual_edit_monitor("sharp ratio")

        # Move to line start with Cmd+Left.
        app._on_global_key_press("Key.cmd")
        app._on_global_key_press("Key.left")
        app._on_global_key_release("Key.cmd")

        # Move to end of "sharp", then type "e" -> "sharpe ratio".
        for _ in range(5):
            app._on_global_key_press("Key.right")
        app._on_global_key_press(FakeCharKey("e"))

        app._finalize_manual_edit_learning()

        app._learner.log_correction.assert_called_once_with("sharp ratio", "sharpe ratio")

    def test_manual_edit_discarded_on_unknown_cmd_shortcut(self, app) -> None:
        app._begin_manual_edit_monitor("sharp ratio")

        app._on_global_key_press("Key.cmd")
        app._on_global_key_press(FakeCharKey("v"))  # Cmd+V
        app._on_global_key_release("Key.cmd")

        assert app._manual_edit_session is None
        app._finalize_manual_edit_learning()
        app._learner.log_correction.assert_not_called()

    def test_manual_edit_reloads_corrector_on_promotion(self, app) -> None:
        import veery.app as app_module

        app._learner.log_correction.return_value = "Sharpe ratio"
        app_module.JargonCorrector.reset_mock()
        app_module.TextCorrector.reset_mock()

        app._begin_manual_edit_monitor("sharp ratio")
        for _ in range(6):
            app._on_global_key_press("Key.left")
        app._on_global_key_press(FakeCharKey("e"))  # sharp -> sharpe

        app._finalize_manual_edit_learning()

        app_module.JargonCorrector.assert_called_once_with(app._config.jargon)
        app_module.TextCorrector.assert_called_once()

    def test_manual_edit_promotion_survives_notification_failure(self, app) -> None:
        app._learner.log_correction.return_value = "Sharpe ratio"
        app._begin_manual_edit_monitor("sharp ratio")
        for _ in range(6):
            app._on_global_key_press("Key.left")
        app._on_global_key_press(FakeCharKey("e"))

        with patch(
            "veery.app.rumps.notification",
            side_effect=RuntimeError("notification unavailable"),
        ):
            app._finalize_manual_edit_learning()

        app._learner.log_correction.assert_called_once_with("sharp ratio", "sharpe ratio")


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

    def test_build_whisper_jargon_prompt_uses_active_terms(self, app) -> None:
        app._config = AppConfig(
            stt=STTConfig(
                whisper_prompt_terms_limit=2,
                whisper_prompt_char_limit=120,
            )
        )
        app._corrector.jargon.dictionary.canonical_terms = ("Veery", "Voiceflow", "PyTorch")

        prompt = app._build_whisper_jargon_prompt()

        assert prompt == "Technical dictation. Prefer these exact spellings: Veery, Voiceflow."

    def test_cleanup_pending_stt_resources_releases_models(self, app) -> None:
        old_stt = MagicMock()
        app._pending_stt_cleanup = [old_stt]

        app._cleanup_pending_stt_resources()

        old_stt.release_resources.assert_called_once()


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

    def test_quit_releases_current_stt(self, app) -> None:
        old_stt = MagicMock()
        app._stt = old_stt
        app._hotkey_listener = None

        with patch("veery.app.rumps"):
            app._on_quit(None)

        old_stt.release_resources.assert_called_once()


# ---------------------------------------------------------------------------
# STT backend switching
# ---------------------------------------------------------------------------


class TestSTTBackendSwitch:
    def test_cancelled_background_whisper_load_keeps_current_stt(self, app) -> None:
        old_stt = MagicMock(name="old_stt")
        new_stt = MagicMock(name="new_stt")
        app._stt = old_stt
        app._whisper_download_cancelled = True

        with (
            patch("veery.app.ensure_model_downloaded") as ensure_downloaded,
            patch("veery.app.create_stt", return_value=new_stt),
            patch.object(app, "_queue_stt_cleanup") as queue_cleanup,
        ):
            app._load_whisper_background()

        ensure_downloaded.assert_called_once()
        assert app._stt is old_stt
        queue_cleanup.assert_called_once_with(new_stt)

    def test_active_background_whisper_load_swaps_and_cleans_up_old_stt(self, app) -> None:
        old_stt = MagicMock(name="old_stt")
        new_stt = MagicMock(name="new_stt")
        app._stt = old_stt
        app._whisper_download_cancelled = False

        with (
            patch("veery.app.ensure_model_downloaded"),
            patch("veery.app.create_stt", return_value=new_stt),
            patch.object(app, "_queue_stt_cleanup") as queue_cleanup,
        ):
            app._load_whisper_background()

        assert app._stt is new_stt
        queue_cleanup.assert_called_once_with(old_stt)

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

    def test_switch_failure_keeps_current_stt_and_releases_switch_claim(self, app) -> None:
        class InlineThread:
            def __init__(self, *, target, daemon) -> None:
                self._target = target

            def start(self) -> None:
                self._target()

        old_stt = MagicMock(name="old_stt")
        app._stt = old_stt
        app._stt_backend = "sensevoice"

        with (
            patch("veery.app.threading.Thread", InlineThread),
            patch("veery.app.create_stt", side_effect=RuntimeError("load failed")),
        ):
            app._on_select_stt_backend("whisper", None)

        assert app._stt is old_stt
        assert app._stt_switching is False
        assert "switch failed" in app._detail_item.title.lower()


from veery.app import State  # noqa: E402  (appended test section)


class TestStreamingReleasePath:
    """Streaming-mode _stop_recording / _process_streaming_release flow."""

    def test_stop_recording_uses_streaming_branch(self, app) -> None:
        app._state = State.RECORDING
        session = MagicMock()
        app._streaming_session = session
        with patch("veery.app.threading.Thread") as thread:
            app._stop_recording()
        app._recorder.stop_capture.assert_called_once()
        assert app._streaming_session is None
        targets = [c.kwargs.get("target") for c in thread.call_args_list]
        assert app._process_streaming_release in targets
        assert app._watch_processing in targets
        assert app._state == State.PROCESSING

    def test_streaming_release_splices_committed_and_pastes(self, app) -> None:
        app._state = State.PROCESSING
        app._corrector = None
        session = MagicMock()
        session.drain.return_value = True
        session.committed.return_value = (["hello world", "第二段"], 42)
        app._recorder.build_tail.return_value = None
        with patch("veery.app.paste_to_active_app") as paste:
            app._process_streaming_release(session, 1)
        app._recorder.build_tail.assert_called_once_with(42)
        paste.assert_called_once()
        assert paste.call_args.args[0] == "hello world第二段"
        assert app._state == State.IDLE

    def test_streaming_release_appends_tail_text(self, app) -> None:
        app._state = State.PROCESSING
        app._corrector = None
        session = MagicMock()
        session.drain.return_value = True
        session.committed.return_value = (["first part"], 10)
        tail = FakeSegment(
            audio=np.full(16000, 0.1, dtype=np.float32), sample_rate=16000, duration_sec=1.0
        )
        app._recorder.build_tail.return_value = tail
        app._stt.transcribe.return_value = "and the tail"
        with patch("veery.app.paste_to_active_app") as paste:
            app._process_streaming_release(session, 1)
        assert paste.call_args.args[0] == "first part and the tail"
        assert app._state == State.IDLE

    def test_tail_stt_error_still_pastes_committed(self, app) -> None:
        app._state = State.PROCESSING
        app._corrector = None
        session = MagicMock()
        session.drain.return_value = True
        session.committed.return_value = (["only part"], 7)
        tail = FakeSegment(
            audio=np.full(16000, 0.1, dtype=np.float32), sample_rate=16000, duration_sec=2.0
        )
        app._recorder.build_tail.return_value = tail
        from veery.stt import STTError

        app._stt.transcribe.side_effect = STTError("backend died")
        with patch("veery.app.paste_to_active_app") as paste:
            app._process_streaming_release(session, 1)
        assert paste.call_args.args[0] == "only part"
        assert app._state == State.IDLE

    def test_drain_timeout_still_pastes_committed(self, app) -> None:
        app._state = State.PROCESSING
        app._corrector = None
        session = MagicMock()
        session.drain.return_value = False  # worker hung
        session.committed.return_value = (["committed before hang"], 5)
        app._recorder.build_tail.return_value = None
        with patch("veery.app.paste_to_active_app") as paste:
            app._process_streaming_release(session, 1)
        assert paste.call_args.args[0] == "committed before hang"

    def test_no_committed_no_tail_warns_no_speech(self, app) -> None:
        app._state = State.PROCESSING
        session = MagicMock()
        session.drain.return_value = True
        session.committed.return_value = ([], 0)
        app._recorder.build_tail.return_value = None
        with patch("veery.app.paste_to_active_app") as paste:
            app._process_streaming_release(session, 1)
        assert not paste.called
        app._overlay.show_warning.assert_called_with("No speech detected")
        assert app._state == State.IDLE

    def test_begin_recording_creates_session_when_enabled(self, app) -> None:
        from dataclasses import replace as dc_replace

        from veery.app import _StreamingSession
        from veery.config import StreamingConfig

        app._config = dc_replace(app._config, streaming=StreamingConfig(enabled=True))
        app._state = State.IDLE
        app._begin_recording()
        assert isinstance(app._streaming_session, _StreamingSession)
        app._recorder.set_finalize_callback.assert_called_with(
            app._streaming_session.enqueue
        )

    def test_begin_recording_no_session_when_disabled(self, app) -> None:
        app._state = State.IDLE
        app._begin_recording()
        assert app._streaming_session is None


# ---------------------------------------------------------------------------
# Edit-capture quality (label pollution guards)
# ---------------------------------------------------------------------------


class TestEditCaptureQuality:
    def test_suppression_window_ignores_synthetic_keys(self, app) -> None:
        """Our own CGEvent-typed output must not count as manual edits."""
        app._begin_manual_edit_monitor("sharp ratio")
        app._suppress_edits_until = time.monotonic() + 5.0
        for ch in " and more":
            app._on_global_key_press(FakeCharKey(ch))
        assert app._manual_edit_session.edit_count == 0
        app._suppress_edits_until = 0.0
        app._finalize_manual_edit_learning()
        app._learner.log_correction.assert_not_called()

    def test_append_only_typing_is_not_a_correction(self, app) -> None:
        """New content typed after the paste is writing, not a correction."""
        app._begin_manual_edit_monitor("sharp ratio")
        for ch in " is high.":
            app._on_global_key_press(FakeCharKey(ch))
        app._finalize_manual_edit_learning()
        app._learner.log_correction.assert_not_called()

    def test_ime_residue_discards_session(self, app) -> None:
        """Raw pinyin in the reconstructed buffer must not become a label."""
        app._begin_manual_edit_monitor("算一下我们平均的市场参与率")
        with app._manual_edit_lock:
            app._manual_edit_session.text = "算一下我们平均的市场jiaoyil 参与率"
            app._manual_edit_session.edit_count = 7
        app._finalize_manual_edit_learning()
        app._learner.log_correction.assert_not_called()

    def test_enter_finalizes_without_newline(self, app) -> None:
        """Enter submits: finalize with the pre-Enter buffer, no '\\n' insertion."""
        app._learner.log_correction.return_value = None
        app._begin_manual_edit_monitor("sharp ratio")
        app._on_global_key_press("Key.cmd")
        app._on_global_key_press("Key.left")
        app._on_global_key_release("Key.cmd")
        for _ in range(5):
            app._on_global_key_press("Key.right")
        app._on_global_key_press(FakeCharKey("e"))
        app._on_global_key_press("Key.enter")
        assert app._manual_edit_session is None
        app._learner.log_correction.assert_called_once_with("sharp ratio", "sharpe ratio")

    def test_enter_without_edits_discards(self, app) -> None:
        app._begin_manual_edit_monitor("sharp ratio")
        app._on_global_key_press("Key.enter")
        assert app._manual_edit_session is None
        app._learner.log_correction.assert_not_called()

    def test_tab_discards_session(self, app) -> None:
        app._begin_manual_edit_monitor("sharp ratio")
        app._on_global_key_press(FakeCharKey("e"))
        app._on_global_key_press("Key.tab")
        assert app._manual_edit_session is None
        app._finalize_manual_edit_learning()
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_rejects_low_similarity(self, app) -> None:
        """Consecutive unrelated sentences (~42% similar) must not pair up."""
        app._last_pasted_text = "sharp ratio"
        app._last_pasted_time = time.monotonic()
        with patch("rapidfuzz.fuzz.ratio", return_value=45):
            app._try_auto_learn("totally different sentence")
        app._learner.log_correction.assert_not_called()

    def test_auto_learn_accepts_real_redictation(self, app) -> None:
        """A genuine re-dictation correction (~55% similar) still learns."""
        app._last_pasted_text = "sharp ratio"
        app._last_pasted_time = time.monotonic()
        app._learner.log_correction.return_value = None
        with patch("rapidfuzz.fuzz.ratio", return_value=55):
            app._try_auto_learn("Sharpe ratio")
        app._learner.log_correction.assert_called_once()


class TestImeResidueDetector:
    @pytest.mark.parametrize(
        "original,edited",
        [
            ("算一下我们平均的市场参与率", "算一下我们平均的市场jiaoyil 参与率"),
            ("方便的呢,入宿空房呢", "方便的呢,dushou 空房呢"),
            ("收益的时间戳固定成了", "收益的时间戳固定hao "),
            ("最近短期剩余一半", "最近短期剩余yiban"),
        ],
        ids=["mid-word-pinyin", "space-separated-pinyin", "trailing-pinyin", "retyped-word"],
    )
    def test_detects_real_pollution_cases(self, original: str, edited: str) -> None:
        from veery.app import _contains_ime_residue

        assert _contains_ime_residue(original, edited)

    @pytest.mark.parametrize(
        "original,edited",
        [
            ("按照In the stock market, Algrim and Chris.", "按照In the stock market, Algren and Chris."),
            ("但是呢要有最大的RI", "但是呢要有最大的ROI"),
            ("基本上是8Bips乘以Turnover", "基本上是8Bps乘以Turnover"),
            ("是我USTT的理财利率吗", "是我USDT的理财利率吗"),
            ("讲话也是能够纠正", "降话也是能够纠正"),
        ],
        ids=["latin-name", "acronym", "unit-fix", "ticker", "pure-cjk-edit"],
    )
    def test_keeps_legitimate_corrections(self, original: str, edited: str) -> None:
        from veery.app import _contains_ime_residue

        assert not _contains_ime_residue(original, edited)
