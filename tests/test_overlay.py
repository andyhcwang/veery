"""Tests for the OverlayIndicator (mocked AppKit)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voiceflow.overlay import OverlayIndicator


@pytest.fixture(autouse=True)
def _reset_overlay_module():
    """Reset module-level state between tests."""
    import voiceflow.overlay as mod
    mod._loaded = False
    mod._overlay_ref = None
    mod._PillView._PillViewClass = None
    yield
    mod._loaded = False
    mod._overlay_ref = None
    mod._PillView._PillViewClass = None


@pytest.fixture
def mock_appkit():
    """Patch _ensure_appkit to succeed without real AppKit."""
    import voiceflow.overlay as mod

    mod._loaded = True
    mod._NSPanel = MagicMock()
    mod._NSColor = MagicMock()
    mod._NSTextField = MagicMock()
    mod._NSFont = MagicMock()
    mod._NSScreen = MagicMock()
    mod._NSView = MagicMock()
    mod._NSBezierPath = MagicMock()
    mod._NSTimer = MagicMock()
    mod._NSAnimationContext = MagicMock()

    return mod


@pytest.fixture
def overlay_with_panel(mock_appkit):
    """Create an overlay with a mocked panel already set up."""
    overlay = OverlayIndicator()
    overlay._panel = MagicMock()
    overlay._label = MagicMock()
    overlay._pill_view = MagicMock()
    return overlay


class TestOverlayIndicator:
    """Tests for OverlayIndicator public API."""

    def test_init_sets_overlay_ref(self):
        """OverlayIndicator.__init__ sets module-level _overlay_ref."""
        import voiceflow.overlay as mod
        overlay = OverlayIndicator()
        assert mod._overlay_ref is overlay

    def test_init_defaults(self):
        """New overlay starts with no panel, no timers."""
        overlay = OverlayIndicator()
        assert overlay._panel is None
        assert overlay._label is None
        assert overlay._pill_view is None
        assert overlay._animation_timer is None
        assert overlay._success_timer is None

    def test_ensure_panel_returns_false_without_appkit(self):
        """_ensure_panel returns False when AppKit is not available."""
        import voiceflow.overlay as mod
        mod._loaded = False
        with patch.object(mod, "_ensure_appkit", return_value=False):
            overlay = OverlayIndicator()
            assert overlay._ensure_panel() is False

    def test_ensure_panel_returns_true_when_panel_exists(self, overlay_with_panel):
        """_ensure_panel returns True if panel already created."""
        assert overlay_with_panel._ensure_panel() is True

    def test_show_recording_dispatches_to_main(self, overlay_with_panel):
        """show_recording dispatches work to main thread via callAfter."""
        with patch("voiceflow.overlay.OverlayIndicator._run_on_main") as mock_run:
            overlay_with_panel.show_recording()
            mock_run.assert_called_once()

    def test_show_processing_dispatches_to_main(self, overlay_with_panel):
        """show_processing dispatches work to main thread via callAfter."""
        with patch("voiceflow.overlay.OverlayIndicator._run_on_main") as mock_run:
            overlay_with_panel.show_processing()
            mock_run.assert_called_once()

    def test_show_success_dispatches_to_main(self, overlay_with_panel):
        """show_success dispatches work to main thread via callAfter."""
        with patch("voiceflow.overlay.OverlayIndicator._run_on_main") as mock_run:
            overlay_with_panel.show_success()
            mock_run.assert_called_once()

    def test_hide_dispatches_to_main(self, overlay_with_panel):
        """hide dispatches work to main thread via callAfter."""
        with patch("voiceflow.overlay.OverlayIndicator._run_on_main") as mock_run:
            overlay_with_panel.hide()
            mock_run.assert_called_once()

    def test_stop_timers_invalidates_animation_timer(self, overlay_with_panel):
        """_stop_timers invalidates and clears animation timer."""
        timer = MagicMock()
        overlay_with_panel._animation_timer = timer
        overlay_with_panel._stop_timers()
        timer.invalidate.assert_called_once()
        assert overlay_with_panel._animation_timer is None

    def test_stop_timers_invalidates_success_timer(self, overlay_with_panel):
        """_stop_timers invalidates and clears success timer."""
        timer = MagicMock()
        overlay_with_panel._success_timer = timer
        overlay_with_panel._stop_timers()
        timer.invalidate.assert_called_once()
        assert overlay_with_panel._success_timer is None

    def test_stop_timers_noop_when_no_timers(self, overlay_with_panel):
        """_stop_timers does nothing when no timers are active."""
        overlay_with_panel._stop_timers()  # should not raise

    def test_show_recording_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes the block synchronously, it updates state."""
        # Make _run_on_main execute the block immediately
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_recording()

        overlay._label.setStringValue_.assert_called_with("Recording")
        overlay._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay._panel.orderFrontRegardless.assert_called()
        assert overlay._pill_view._mode == "recording"

    def test_show_processing_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, processing state is set."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_processing()

        overlay._label.setStringValue_.assert_called_with("Processing")
        overlay._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay._panel.orderFrontRegardless.assert_called()
        assert overlay._pill_view._mode == "processing"

    def test_show_success_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, success state is set."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_success()

        overlay._label.setStringValue_.assert_called_with("Done")
        overlay._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay._panel.orderFrontRegardless.assert_called()
        assert overlay._pill_view._mode is None  # success has no dot mode

    def test_hide_calls_fade_out(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, hide fades out."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        with patch.object(overlay, "_fade_out") as mock_fade:
            overlay.hide()
            mock_fade.assert_called_once()

    def test_show_recording_stops_existing_timers(self, mock_appkit, overlay_with_panel):
        """show_recording stops any running timers before starting new ones."""
        def run_sync(block):
            block()

        old_timer = MagicMock()
        overlay = overlay_with_panel
        overlay._animation_timer = old_timer
        overlay._run_on_main = run_sync
        overlay.show_recording()

        old_timer.invalidate.assert_called_once()

    def test_show_recording_starts_pulse_timer(self, mock_appkit, overlay_with_panel):
        """show_recording starts an NSTimer for pulse animation."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_recording()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.05  # 50ms interval
        assert call_args[0][2] == b"pulseTimer:"
        assert call_args[0][4] is True  # repeats

    def test_show_processing_starts_cycle_timer(self, mock_appkit, overlay_with_panel):
        """show_processing starts an NSTimer for dot cycling."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_processing()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.4  # 400ms interval
        assert call_args[0][2] == b"cycleDotsTimer:"
        assert call_args[0][4] is True  # repeats

    def test_show_success_starts_hide_timer(self, mock_appkit, overlay_with_panel):
        """show_success starts an NSTimer that auto-hides after 800ms."""
        def run_sync(block):
            block()

        overlay = overlay_with_panel
        overlay._run_on_main = run_sync
        overlay.show_success()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.8  # 800ms
        assert call_args[0][2] == b"successHideTimer:"
        assert call_args[0][4] is False  # does not repeat

    def test_run_on_main_handles_import_error(self):
        """_run_on_main logs exception if callAfter import fails."""
        overlay = OverlayIndicator()
        with patch("voiceflow.overlay.logger") as mock_logger:
            with patch.dict("sys.modules", {"PyObjCTools": None, "PyObjCTools.AppHelper": None}):
                overlay._run_on_main(lambda: None)
            mock_logger.exception.assert_called()

    def test_ensure_panel_returns_false_when_no_screen(self, mock_appkit):
        """_ensure_panel returns False if no screen is available."""
        mock_appkit._NSScreen.mainScreen.return_value = None
        overlay = OverlayIndicator()
        assert overlay._ensure_panel() is False

    def test_hide_noop_when_no_panel(self):
        """hide() is safe to call when panel is None."""
        def run_sync(block):
            block()

        overlay = OverlayIndicator()
        overlay._run_on_main = run_sync
        overlay.hide()  # should not raise


class TestOverlayConstants:
    """Verify overlay design constants match spec."""

    def test_pill_dimensions(self):
        from voiceflow.overlay import _PILL_HEIGHT, _PILL_RADIUS, _PILL_WIDTH
        assert _PILL_WIDTH == 180
        assert _PILL_HEIGHT == 44
        assert _PILL_RADIUS == 22

    def test_pill_position(self):
        from voiceflow.overlay import _PILL_TOP_OFFSET
        assert _PILL_TOP_OFFSET == 60

    def test_background_color(self):
        from voiceflow.overlay import _BG_ALPHA, _BG_BLUE, _BG_GREEN, _BG_RED
        assert _BG_RED == 0.08
        assert _BG_GREEN == 0.08
        assert _BG_BLUE == 0.08
        assert _BG_ALPHA == 0.92
