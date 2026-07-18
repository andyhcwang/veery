"""Tests for the OverlayIndicator (mocked AppKit)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from veery.overlay import OverlayIndicator


@pytest.fixture(autouse=True)
def _reset_overlay_module():
    """Reset module-level state between tests."""
    import veery.overlay as mod
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
    import veery.overlay as mod

    mod._loaded = True
    mod._NSPanel = MagicMock()
    mod._NSColor = MagicMock()
    mod._NSTextField = MagicMock()
    mod._NSFont = MagicMock()
    mod._NSScreen = MagicMock()
    mod._NSView = MagicMock()
    mod._NSBezierPath = MagicMock()
    mod._NSTimer = MagicMock()
    mod._NSGlassEffectView = MagicMock()
    mod._NSVisualEffectView = MagicMock()
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


def _patch_run_on_main_sync():
    """Return a patch context that makes _run_on_main execute blocks synchronously."""
    def run_sync(block):
        block()
    return patch("veery.overlay._run_on_main", side_effect=run_sync)


class TestOverlayIndicator:
    """Tests for OverlayIndicator public API."""

    def test_init_sets_overlay_ref(self):
        """OverlayIndicator.__init__ sets module-level _overlay_ref."""
        import veery.overlay as mod
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
        import veery.overlay as mod
        mod._loaded = False
        with patch.object(mod, "_ensure_appkit", return_value=False):
            overlay = OverlayIndicator()
            assert overlay._ensure_panel() is False

    def test_ensure_panel_returns_true_when_panel_exists(self, overlay_with_panel):
        """_ensure_panel returns True if panel already created."""
        assert overlay_with_panel._ensure_panel() is True

    def test_show_recording_dispatches_to_main(self, overlay_with_panel):
        """show_recording dispatches work to main thread via callAfter."""
        with patch("veery.overlay._run_on_main") as mock_run:
            overlay_with_panel.show_recording()
            mock_run.assert_called_once()

    def test_show_processing_dispatches_to_main(self, overlay_with_panel):
        """show_processing dispatches work to main thread via callAfter."""
        with patch("veery.overlay._run_on_main") as mock_run:
            overlay_with_panel.show_processing()
            mock_run.assert_called_once()

    def test_show_success_dispatches_to_main(self, overlay_with_panel):
        """show_success dispatches work to main thread via callAfter."""
        with patch("veery.overlay._run_on_main") as mock_run:
            overlay_with_panel.show_success()
            mock_run.assert_called_once()

    def test_hide_dispatches_to_main(self, overlay_with_panel):
        """hide dispatches work to main thread via callAfter."""
        with patch("veery.overlay._run_on_main") as mock_run:
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
        with _patch_run_on_main_sync():
            overlay_with_panel.show_recording()

        overlay_with_panel._label.setStringValue_.assert_called_with("Listening")
        overlay_with_panel._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay_with_panel._panel.orderFrontRegardless.assert_called()
        assert overlay_with_panel._pill_view._mode == "recording"

    def test_show_processing_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, processing state is set."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_processing()

        overlay_with_panel._label.setStringValue_.assert_called_with("Processing")
        overlay_with_panel._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay_with_panel._panel.orderFrontRegardless.assert_called()
        assert overlay_with_panel._pill_view._mode == "processing"

    def test_show_success_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, success state is set."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_success()

        overlay_with_panel._label.setStringValue_.assert_called_with("Done")
        overlay_with_panel._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay_with_panel._panel.orderFrontRegardless.assert_called()
        assert overlay_with_panel._pill_view._mode == "success"

    def test_hide_calls_fade_out(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, hide fades out."""
        with _patch_run_on_main_sync(), \
             patch.object(overlay_with_panel, "_fade_out") as mock_fade:
            overlay_with_panel.hide()
            mock_fade.assert_called_once()

    def test_show_recording_stops_existing_timers(self, mock_appkit, overlay_with_panel):
        """show_recording stops any running timers before starting new ones."""
        old_timer = MagicMock()
        overlay_with_panel._animation_timer = old_timer
        with _patch_run_on_main_sync():
            overlay_with_panel.show_recording()

        old_timer.invalidate.assert_called_once()

    def test_show_recording_starts_pulse_timer(self, mock_appkit, overlay_with_panel):
        """show_recording starts an NSTimer for pulse animation."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_recording()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.05  # 50ms interval
        assert call_args[0][2] == b"pulseTimer:"
        assert call_args[0][4] is True  # repeats

    def test_show_processing_starts_cycle_timer(self, mock_appkit, overlay_with_panel):
        """show_processing starts an NSTimer for dot cycling."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_processing()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.4  # 400ms interval
        assert call_args[0][2] == b"cycleDotsTimer:"
        assert call_args[0][4] is True  # repeats

    def test_show_success_starts_hide_timer(self, mock_appkit, overlay_with_panel):
        """show_success starts an NSTimer that auto-hides after 1200ms."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_success()

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 1.2  # 1200ms
        assert call_args[0][2] == b"successHideTimer:"
        assert call_args[0][4] is False  # does not repeat

    def test_run_on_main_handles_import_error(self):
        """_run_on_main logs exception if callAfter import fails."""
        from veery.overlay import _run_on_main

        with patch("veery.overlay.logger") as mock_logger:
            with patch.dict("sys.modules", {"PyObjCTools": None, "PyObjCTools.AppHelper": None}):
                _run_on_main(lambda: None)
            mock_logger.exception.assert_called()

    def test_ensure_panel_returns_false_when_no_screen(self, mock_appkit):
        """_ensure_panel returns False if no screen is available."""
        mock_appkit._NSScreen.mainScreen.return_value = None
        overlay = OverlayIndicator()
        assert overlay._ensure_panel() is False

    def test_show_warning_dispatches_to_main(self, overlay_with_panel):
        """show_warning dispatches work to main thread via callAfter."""
        with patch("veery.overlay._run_on_main") as mock_run:
            overlay_with_panel.show_warning("No speech detected")
            mock_run.assert_called_once()

    def test_show_warning_calls_block_directly(self, mock_appkit, overlay_with_panel):
        """When _run_on_main executes synchronously, warning state is set."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_warning("No speech detected")

        overlay_with_panel._label.setStringValue_.assert_called_with("No speech detected")
        overlay_with_panel._pill_view.setNeedsDisplay_.assert_called_with(True)
        overlay_with_panel._panel.orderFrontRegardless.assert_called()
        assert overlay_with_panel._pill_view._mode is None  # no dot for warnings

    def test_show_warning_starts_auto_hide_timer(self, mock_appkit, overlay_with_panel):
        """show_warning starts an NSTimer that auto-hides after 1.5s."""
        with _patch_run_on_main_sync():
            overlay_with_panel.show_warning("Test warning")

        mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = mock_appkit._NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 1.5  # 1500ms
        assert call_args[0][2] == b"successHideTimer:"
        assert call_args[0][4] is False  # does not repeat

    def test_hide_noop_when_no_panel(self):
        """hide() is safe to call when panel is None."""
        overlay = OverlayIndicator()
        with _patch_run_on_main_sync():
            overlay.hide()  # should not raise


class TestPermissionDependencies:
    def test_avfoundation_available_for_microphone_permission_flow(self):
        """Microphone permission checks need AVFoundation at runtime."""
        importlib.import_module("AVFoundation")


class TestOverlayConstants:
    """Verify overlay design constants match spec."""

    def test_pill_dimensions(self):
        from veery.overlay import _PILL_HEIGHT, _PILL_RADIUS, _PILL_WIDTH
        assert _PILL_WIDTH == 180
        assert _PILL_HEIGHT == 44
        assert _PILL_RADIUS == 22

    def test_pill_position(self):
        from veery.overlay import _PILL_TOP_OFFSET
        assert _PILL_TOP_OFFSET == 60

    def test_background_color(self):
        from veery.overlay import _BG_ALPHA, _BG_BLUE, _BG_GREEN, _BG_RED
        assert _BG_RED == 0.08
        assert _BG_GREEN == 0.08
        assert _BG_BLUE == 0.08
        assert _BG_ALPHA == 0.78


class TestFreshPermissionCheck:
    """_fresh_permission_check re-evaluates TCC in a child process."""

    def test_unknown_kind_returns_false(self):
        from veery.overlay import _fresh_permission_check

        assert _fresh_permission_check("nonsense") is False

    def test_granted_child_exit_zero(self):
        from veery.overlay import _fresh_permission_check

        proc = MagicMock(returncode=0)
        with patch("veery.overlay.subprocess.run", return_value=proc) as run:
            assert _fresh_permission_check("accessibility") is True
        assert run.called

    def test_denied_child_exit_nonzero(self):
        from veery.overlay import _fresh_permission_check

        proc = MagicMock(returncode=1)
        with patch("veery.overlay.subprocess.run", return_value=proc):
            assert _fresh_permission_check("input_monitoring") is False

    def test_child_failure_returns_false(self):
        from veery.overlay import _fresh_permission_check

        with patch("veery.overlay.subprocess.run", side_effect=OSError("spawn failed")):
            assert _fresh_permission_check("accessibility") is False


class TestStaleGrantRelaunch:
    """Granted-but-cached-denied permissions trigger the relaunch path."""

    def _make_overlay(self):
        from veery.overlay import PermissionGuideOverlay

        overlay = PermissionGuideOverlay()
        overlay._pending_steps = [0]
        overlay._current_step = 0
        return overlay

    def test_stale_grant_detected_every_third_poll(self):
        overlay = self._make_overlay()
        with (
            patch("veery.overlay._check_accessibility", return_value=False),
            patch("veery.overlay._fresh_permission_check", return_value=True) as fresh,
            patch.object(overlay, "_handle_stale_grant") as handle,
        ):
            overlay._poll_permission()
            overlay._poll_permission()
            assert not fresh.called  # polls 1-2: no child spawned
            overlay._poll_permission()  # poll 3: fresh check runs
        assert fresh.call_count == 1
        assert handle.call_count == 1

    def test_fresh_check_denied_keeps_polling(self):
        overlay = self._make_overlay()
        with (
            patch("veery.overlay._check_accessibility", return_value=False),
            patch("veery.overlay._fresh_permission_check", return_value=False),
            patch.object(overlay, "_handle_stale_grant") as handle,
        ):
            for _ in range(6):
                overlay._poll_permission()
        assert not handle.called

    def test_microphone_step_never_spawns_child(self):
        overlay = self._make_overlay()
        overlay._pending_steps = [1]
        with (
            patch("veery.overlay._check_microphone", return_value=False),
            patch("veery.overlay._fresh_permission_check") as fresh,
        ):
            for _ in range(6):
                overlay._poll_permission()
        assert not fresh.called

    def test_normal_grant_advances_without_child(self):
        overlay = self._make_overlay()
        overlay._on_complete = MagicMock()
        with (
            patch("veery.overlay._check_accessibility", return_value=True),
            patch("veery.overlay._fresh_permission_check") as fresh,
            patch.object(overlay, "_stop_timer"),
            patch.object(overlay, "_fade_out"),
        ):
            overlay._poll_permission()
        assert not fresh.called
        overlay._on_complete.assert_called_once()

    def test_handle_stale_grant_dev_mode_instructs_restart(self):
        import sys

        overlay = self._make_overlay()
        overlay._body_label = MagicMock()
        fake_appkit = MagicMock()
        fake_appkit.NSBundle.mainBundle.return_value.bundlePath.return_value = (
            "/Users/x/checkout"  # not a .app bundle -> dev mode
        )
        with (
            patch.dict(sys.modules, {"AppKit": fake_appkit}),
            patch.object(overlay, "_stop_timer") as stop,
            patch("veery.overlay.subprocess.Popen") as popen,
        ):
            overlay._handle_stale_grant()
        stop.assert_called_once()
        assert not popen.called  # no relaunch in dev mode
        msg = overlay._body_label.setStringValue_.call_args[0][0]
        assert "restart" in msg.lower()

    def test_handle_stale_grant_bundle_relaunches(self):
        import sys

        overlay = self._make_overlay()
        overlay._body_label = MagicMock()
        fake_appkit = MagicMock()
        fake_appkit.NSBundle.mainBundle.return_value.bundlePath.return_value = (
            "/Applications/Veery.app"
        )
        with (
            patch.dict(sys.modules, {"AppKit": fake_appkit}),
            patch.object(overlay, "_stop_timer"),
            patch("veery.overlay.subprocess.Popen") as popen,
        ):
            overlay._handle_stale_grant()
        assert popen.called
        assert "/Applications/Veery.app" in popen.call_args[0][0][-1]
        fake_appkit.NSApplication.sharedApplication.return_value.terminate_.assert_called_once()


class TestPermissionPrompts:
    """Step transitions fire the matching system permission prompt."""

    def test_step_dispatch(self):
        import veery.overlay as mod

        with (
            patch.object(mod, "_request_accessibility") as ax,
            patch.object(mod, "_request_microphone") as mic,
            patch.object(mod, "_request_input_monitoring") as im,
        ):
            mod._request_permission_for_step(0)
            mod._request_permission_for_step(1)
            mod._request_permission_for_step(2)
        ax.assert_called_once()
        mic.assert_called_once()
        im.assert_called_once()

    def test_poll_advance_requests_next_permission(self):
        from veery.overlay import PermissionGuideOverlay

        overlay = PermissionGuideOverlay()
        overlay._pending_steps = [0, 2]
        overlay._current_step = 0
        with (
            patch("veery.overlay._check_accessibility", return_value=True),
            patch.object(overlay, "_update_step_content"),
            patch.object(overlay, "_open_settings"),
            patch("veery.overlay._request_permission_for_step") as req,
        ):
            overlay._poll_permission()
        req.assert_called_once_with(2)

    def test_request_helpers_never_raise(self):
        import veery.overlay as mod

        with patch("veery.overlay.ctypes.cdll.LoadLibrary", side_effect=OSError("no lib")):
            mod._request_input_monitoring()  # must not raise
        import sys

        with patch.dict(sys.modules, {"ApplicationServices": None}):
            mod._request_accessibility()  # ImportError path must not raise
