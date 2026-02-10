"""Floating overlay indicator for recording/processing state.

Shows a polished translucent pill at the top-center of the screen that
doesn't steal focus from the active application. Features animated
pulsing dot (recording), cycling ellipsis (processing), and brief
success flash.

Also provides PermissionGuideOverlay for first-launch onboarding.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import subprocess
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _run_on_main(block) -> None:
    """Dispatch a block to the main thread."""
    try:
        from PyObjCTools.AppHelper import callAfter
        callAfter(block)
    except Exception:
        logger.exception("Failed to dispatch to main thread")


def _get_active_screen() -> object:
    """Get the screen where the user is currently working (follows mouse cursor).

    Falls back to the main screen if the mouse is not found on any screen.
    """
    try:
        import AppKit

        mouse_loc = AppKit.NSEvent.mouseLocation()
        screens = _NSScreen.screens()
        if screens is None:
            return _NSScreen.mainScreen()

        for screen in screens:
            frame = screen.frame()
            if (frame.origin.x <= mouse_loc.x < frame.origin.x + frame.size.width and
                    frame.origin.y <= mouse_loc.y < frame.origin.y + frame.size.height):
                return screen

        logger.debug(
            "Mouse at (%.0f, %.0f) not within any of %d screens, falling back to main",
            mouse_loc.x, mouse_loc.y, len(screens),
        )
    except Exception:
        logger.debug("Failed to determine active screen, falling back to main screen")

    return _NSScreen.mainScreen()


# Lazy-loaded AppKit/Quartz references
_NSPanel = None
_NSColor = None
_NSTextField = None
_NSFont = None
_NSScreen = None
_NSView = None
_NSBezierPath = None
_NSTimer = None
_NSGlassEffectView = None  # macOS Tahoe+ (Liquid Glass)
_NSVisualEffectView = None  # fallback for older macOS
_NSAnimationContext = None
_loaded = False

# Pill dimensions and position
_PILL_WIDTH = 180
_PILL_HEIGHT = 44
_PILL_RADIUS = 22
_PILL_TOP_OFFSET = 60

# Colors
_BG_RED = 0.08
_BG_GREEN = 0.08
_BG_BLUE = 0.08
_BG_ALPHA = 0.78

_RECORDING_DOT_RED = 1.0
_RECORDING_DOT_GREEN = 0.231
_RECORDING_DOT_BLUE = 0.188

_SUCCESS_GREEN_RED = 0.35
_SUCCESS_GREEN_GREEN = 0.82
_SUCCESS_GREEN_BLUE = 0.60

# Module-level back-reference so the ObjC view can call overlay.hide()
_overlay_ref = None  # type: OverlayIndicator | None


def _ensure_appkit() -> bool:
    """Lazy-load AppKit classes. Returns True if available."""
    global _NSPanel, _NSColor, _NSTextField, _NSFont, _NSScreen, _NSView
    global _NSBezierPath, _NSTimer, _NSGlassEffectView, _NSVisualEffectView
    global _NSAnimationContext, _loaded
    if _loaded:
        return True
    try:
        import AppKit

        _NSPanel = AppKit.NSPanel
        _NSColor = AppKit.NSColor
        _NSTextField = AppKit.NSTextField
        _NSFont = AppKit.NSFont
        _NSScreen = AppKit.NSScreen
        _NSView = AppKit.NSView
        _NSBezierPath = AppKit.NSBezierPath
        _NSTimer = AppKit.NSTimer
        _NSGlassEffectView = getattr(AppKit, "NSGlassEffectView", None)
        _NSVisualEffectView = AppKit.NSVisualEffectView
        _NSAnimationContext = AppKit.NSAnimationContext
        _loaded = True
        return True
    except ImportError:
        logger.warning("AppKit not available, overlay disabled")
        return False


# Style masks for a borderless, non-activating panel
_BORDERLESS = 0
_NON_ACTIVATING = 1 << 7  # NSWindowStyleMaskNonactivatingPanel


class _PillView:
    """Custom NSView subclass that draws a rounded-rect pill with animations."""

    _PillViewClass = None

    @classmethod
    def get_class(cls):
        if cls._PillViewClass is not None:
            return cls._PillViewClass

        import objc  # noqa: F401

        NSView = _NSView

        class PillBackgroundView(NSView):
            """Pill background with animated recording dot or processing dots.

            Timer callbacks are defined here as proper ObjC methods so NSTimer
            can invoke them via selectors.
            """

            def initWithFrame_(self, frame):
                self = objc.super(PillBackgroundView, self).initWithFrame_(frame)
                if self is not None:
                    self._mode = None
                    self._dot_alpha = 1.0
                    self._pulse_dir = -1
                    self._dot_phase = 0
                    self._bg_red = _BG_RED
                    self._bg_green = _BG_GREEN
                    self._bg_blue = _BG_BLUE
                    self._has_blur = False
                return self

            def drawRect_(self, rect):
                if not self._has_blur:
                    _NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        self._bg_red, self._bg_green, self._bg_blue, _BG_ALPHA
                    ).set()
                    path = _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        rect, _PILL_RADIUS, _PILL_RADIUS
                    )
                    path.fill()

                if self._mode == "recording":
                    self._draw_recording_dot(rect)
                elif self._mode == "processing":
                    self._draw_processing_dots(rect)
                elif self._mode == "success":
                    self._draw_success_dot(rect)

            def _draw_dot(self, rect, r, g, b, a):
                """Draw a colored dot on the left side of the pill."""
                dot_size = 8
                dot_x = 18
                dot_y = (rect.size.height - dot_size) / 2
                dot_rect = ((dot_x, dot_y), (dot_size, dot_size))
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).set()
                _NSBezierPath.bezierPathWithOvalInRect_(dot_rect).fill()

            def _draw_recording_dot(self, rect):
                """Draw a pulsing red dot on the left side."""
                self._draw_dot(
                    rect,
                    _RECORDING_DOT_RED, _RECORDING_DOT_GREEN, _RECORDING_DOT_BLUE,
                    self._dot_alpha,
                )

            def _draw_processing_dots(self, rect):
                """Draw a single white dot on the left side during processing."""
                self._draw_dot(rect, 1.0, 1.0, 1.0, 0.7)

            def _draw_success_dot(self, rect):
                """Draw a green dot on the left side for success state."""
                self._draw_dot(
                    rect,
                    _SUCCESS_GREEN_RED, _SUCCESS_GREEN_GREEN, _SUCCESS_GREEN_BLUE,
                    1.0,
                )

            # --- NSTimer callbacks (ObjC selectors) ---

            def pulseTimer_(self, timer):
                """Animate recording dot alpha between 0.5 and 1.0."""
                step = 0.033
                self._dot_alpha += step * self._pulse_dir
                if self._dot_alpha <= 0.5:
                    self._dot_alpha = 0.5
                    self._pulse_dir = 1
                elif self._dot_alpha >= 1.0:
                    self._dot_alpha = 1.0
                    self._pulse_dir = -1
                self.setNeedsDisplay_(True)

            def cycleDotsTimer_(self, timer):
                """Animate 'Processing' text with cycling ellipsis."""
                self._dot_phase = (self._dot_phase + 1) % 4
                dots = "." * self._dot_phase
                if _overlay_ref is not None and _overlay_ref._label is not None:
                    _overlay_ref._label.setStringValue_(f"Processing{dots}")

            def successHideTimer_(self, timer):
                """Auto-hide overlay after success flash."""
                if _overlay_ref is not None:
                    _overlay_ref.hide()

        cls._PillViewClass = PillBackgroundView
        return cls._PillViewClass


class OverlayIndicator:
    """Floating pill overlay that shows recording/processing/success status.

    Usage:
        overlay = OverlayIndicator()
        overlay.show_recording()   # Shows pulsing red dot + "Recording"
        overlay.show_processing()  # Shows animated dots + "Processing"
        overlay.show_success()     # Brief green flash + "Done", then auto-hides
        overlay.hide()             # Hides the overlay
    """

    def __init__(self) -> None:
        global _overlay_ref
        self._panel = None
        self._label = None
        self._pill_view = None
        self._lock = threading.Lock()
        self._animation_timer = None
        self._success_timer = None
        _overlay_ref = self

    def _ensure_panel(self) -> bool:
        """Create the panel lazily on first use. Must be called from main thread."""
        if self._panel is not None:
            return True

        if not _ensure_appkit():
            return False

        try:
            import AppKit  # noqa: F401

            screen = _get_active_screen()
            if screen is None:
                return False
            screen_frame = screen.frame()

            x = screen_frame.origin.x + (screen_frame.size.width - _PILL_WIDTH) / 2
            y = screen_frame.origin.y + screen_frame.size.height - _PILL_HEIGHT - _PILL_TOP_OFFSET
            frame = ((x, y), (_PILL_WIDTH, _PILL_HEIGHT))

            panel = _NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                _BORDERLESS | _NON_ACTIVATING,
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(25)  # NSStatusWindowLevel
            panel.setOpaque_(False)
            panel.setBackgroundColor_(_NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces
            panel.setAlphaValue_(0.0)

            # Blur background — NSVisualEffectView with hudWindow material.
            # NSGlassEffectView (Liquid Glass) is available on Tahoe but requires
            # vibrancy-aware text rendering for legibility; revisit when adopting
            # NSTextField vibrancy or SwiftUI bridging.
            has_blur = False
            if _NSVisualEffectView is not None:
                try:
                    blur_view = _NSVisualEffectView.alloc().initWithFrame_(
                        ((0, 0), (_PILL_WIDTH, _PILL_HEIGHT))
                    )
                    blur_view.setBlendingMode_(1)  # behindWindow
                    blur_view.setState_(1)  # active
                    blur_view.setMaterial_(12)  # popover (lighter than hudWindow)
                    blur_view.setWantsLayer_(True)
                    blur_view.layer().setCornerRadius_(_PILL_RADIUS)
                    blur_view.layer().setMasksToBounds_(True)
                    dark = AppKit.NSAppearance.appearanceNamed_(
                        "NSAppearanceNameVibrantDark"
                    )
                    if dark is not None:
                        blur_view.setAppearance_(dark)
                    blur_view.setAlphaValue_(0.78)
                    panel.contentView().addSubview_(blur_view)
                    has_blur = True
                    logger.debug("Using NSVisualEffectView fallback")
                except Exception:
                    logger.debug("NSVisualEffectView unavailable, using solid background")

            # Pill overlay view (draws animated dots on transparent background)
            PillView = _PillView.get_class()
            pill_view = PillView.alloc().initWithFrame_(
                ((0, 0), (_PILL_WIDTH, _PILL_HEIGHT))
            )
            pill_view._has_blur = has_blur
            panel.contentView().addSubview_(pill_view)

            # Text label — vertically centered, offset right for dot area
            label_x = 34
            label_width = _PILL_WIDTH - label_x - 14
            label_height = 20
            label_y = (_PILL_HEIGHT - label_height) / 2
            label = _NSTextField.alloc().initWithFrame_(
                ((label_x, label_y), (label_width, label_height))
            )
            label.setStringValue_("")
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setAlignment_(0)  # NSTextAlignmentLeft
            label.setTextColor_(_NSColor.whiteColor())
            label.setFont_(_NSFont.systemFontOfSize_weight_(13, 0.3))  # Semibold
            panel.contentView().addSubview_(label)

            self._panel = panel
            self._label = label
            self._pill_view = pill_view
            return True

        except Exception:
            logger.exception("Failed to create overlay panel")
            return False

    def _stop_timers(self) -> None:
        """Invalidate any running animation timers. Must be called from main thread."""
        if self._animation_timer is not None:
            self._animation_timer.invalidate()
            self._animation_timer = None
        if self._success_timer is not None:
            self._success_timer.invalidate()
            self._success_timer = None

    def _reposition_panel(self) -> None:
        """Update panel position to follow the active screen. Must be called from main thread."""
        if self._panel is None:
            return
        screen = _get_active_screen()
        if screen is None:
            return
        screen_frame = screen.frame()
        x = screen_frame.origin.x + (screen_frame.size.width - _PILL_WIDTH) / 2
        y = screen_frame.origin.y + screen_frame.size.height - _PILL_HEIGHT - _PILL_TOP_OFFSET
        self._panel.setFrameOrigin_((x, y))

    def _fade_in(self) -> None:
        """Fade the panel in. Must be called from main thread."""
        if self._panel is None:
            return
        # Reposition to follow active screen before showing
        self._reposition_panel()
        _NSAnimationContext.beginGrouping()
        _NSAnimationContext.currentContext().setDuration_(0.15)
        self._panel.animator().setAlphaValue_(1.0)
        _NSAnimationContext.endGrouping()

    def _fade_out(self) -> None:
        """Fade the panel out. Must be called from main thread."""
        if self._panel is None:
            return
        _NSAnimationContext.beginGrouping()
        _NSAnimationContext.currentContext().setDuration_(0.2)
        self._panel.animator().setAlphaValue_(0.0)
        _NSAnimationContext.endGrouping()

    def show_recording(self) -> None:
        """Show the recording indicator with pulsing red dot."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._stop_timers()
                self._pill_view._mode = "recording"
                self._pill_view._dot_alpha = 1.0
                self._pill_view._pulse_dir = -1
                self._pill_view._bg_red = _BG_RED
                self._pill_view._bg_green = _BG_GREEN
                self._pill_view._bg_blue = _BG_BLUE
                self._label.setTextColor_(_NSColor.whiteColor())
                self._label.setStringValue_("Listening")
                self._pill_view.setNeedsDisplay_(True)
                self._panel.orderFrontRegardless()
                self._fade_in()

                self._animation_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        0.05, self._pill_view, b"pulseTimer:", None, True
                    )
                )

        _run_on_main(_show)

    def show_processing(self) -> None:
        """Show the processing indicator with animated dots."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._stop_timers()
                self._pill_view._mode = "processing"
                self._pill_view._dot_phase = 0
                self._pill_view._bg_red = _BG_RED
                self._pill_view._bg_green = _BG_GREEN
                self._pill_view._bg_blue = _BG_BLUE
                self._label.setTextColor_(_NSColor.whiteColor())
                self._label.setStringValue_("Processing")
                self._pill_view.setNeedsDisplay_(True)
                self._panel.orderFrontRegardless()
                self._fade_in()

                self._animation_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        0.4, self._pill_view, b"cycleDotsTimer:", None, True
                    )
                )

        _run_on_main(_show)

    def show_success(self) -> None:
        """Show brief success flash, then auto-hide after ~1.2s."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._stop_timers()
                self._pill_view._mode = "success"
                self._pill_view._bg_red = _BG_RED
                self._pill_view._bg_green = _BG_GREEN
                self._pill_view._bg_blue = _BG_BLUE
                self._label.setTextColor_(_NSColor.whiteColor())
                self._label.setStringValue_("Done")
                self._pill_view.setNeedsDisplay_(True)
                self._panel.orderFrontRegardless()
                self._fade_in()

                self._success_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        1.2, self._pill_view, b"successHideTimer:", None, False
                    )
                )

        _run_on_main(_show)

    def show_warning(self, message: str) -> None:
        """Show a brief warning message in the pill, then auto-hide after ~1.5s."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._stop_timers()
                self._pill_view._mode = None  # no dot
                self._pill_view._bg_red = _BG_RED
                self._pill_view._bg_green = _BG_GREEN
                self._pill_view._bg_blue = _BG_BLUE
                self._label.setTextColor_(
                    _NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.6)
                )
                self._label.setStringValue_(message)
                self._pill_view.setNeedsDisplay_(True)
                self._panel.orderFrontRegardless()
                self._fade_in()

                self._success_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        1.5, self._pill_view, b"successHideTimer:", None, False
                    )
                )

        _run_on_main(_show)

    def hide(self) -> None:
        """Hide the overlay."""
        def _hide():
            with self._lock:
                self._stop_timers()
                if self._panel is not None:
                    self._fade_out()

        _run_on_main(_hide)


# ---- Download progress overlay ----

# Download overlay dimensions
_DL_WIDTH = 400
_DL_HEIGHT = 180
_DL_RADIUS = 20

# Progress bar colors
_PROGRESS_BLUE_RED = 0.29
_PROGRESS_BLUE_GREEN = 0.62
_PROGRESS_BLUE_BLUE = 1.0
_PROGRESS_TRACK_ALPHA = 0.15
_TIPS_ALPHA = 0.4

_TIPS = (
    "Your speech models run 100% locally \u2014 nothing leaves your Mac",
    "Hold Right \u2318 to dictate, release to process",
    "Veery learns your jargon over time",
    "Supports bilingual Chinese/English dictation",
)


class _DownloadPanelView:
    """Custom NSView subclass that draws the download progress panel background."""

    _ViewClass = None

    @classmethod
    def get_class(cls):
        if cls._ViewClass is not None:
            return cls._ViewClass

        import objc  # noqa: F401

        NSView = _NSView

        class DownloadBackgroundView(NSView):
            """Rounded-rect background for the download overlay."""

            def drawRect_(self, rect):
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    _BG_RED, _BG_GREEN, _BG_BLUE, _BG_ALPHA
                ).set()
                path = _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, _DL_RADIUS, _DL_RADIUS
                )
                path.fill()

        cls._ViewClass = DownloadBackgroundView
        return cls._ViewClass


class _ProgressBarView:
    """Custom NSView subclass for the thin progress bar with track and fill."""

    _ViewClass = None

    @classmethod
    def get_class(cls):
        if cls._ViewClass is not None:
            return cls._ViewClass

        import objc  # noqa: F401

        NSView = _NSView

        class ProgressBarDrawView(NSView):
            """Draws a thin rounded progress bar."""

            def initWithFrame_(self, frame):
                self = objc.super(ProgressBarDrawView, self).initWithFrame_(frame)
                if self is not None:
                    self._fraction = 0.0
                return self

            def drawRect_(self, rect):
                bar_height = 4
                bar_y = (rect.size.height - bar_height) / 2
                bar_rect = ((0, bar_y), (rect.size.width, bar_height))

                # Track
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, _PROGRESS_TRACK_ALPHA
                ).set()
                _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    bar_rect, bar_height / 2, bar_height / 2
                ).fill()

                # Fill
                if self._fraction > 0:
                    fill_width = rect.size.width * min(self._fraction, 1.0)
                    fill_rect = ((0, bar_y), (fill_width, bar_height))
                    _NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        _PROGRESS_BLUE_RED,
                        _PROGRESS_BLUE_GREEN,
                        _PROGRESS_BLUE_BLUE,
                        1.0,
                    ).set()
                    _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        fill_rect, bar_height / 2, bar_height / 2
                    ).fill()

            def cycleTipsTimer_(self, timer):
                """Advance tips text. Called by NSTimer via selector."""
                pass  # handled by DownloadProgressOverlay

        cls._ViewClass = ProgressBarDrawView
        return cls._ViewClass


class DownloadProgressOverlay:
    """Larger overlay panel showing model download progress with a progress bar and tips.

    Usage:
        overlay = DownloadProgressOverlay()
        overlay.show()
        overlay.set_progress(0.5, "Downloading SenseVoice-Small (500 MB / 1.0 GB)...")
        overlay.hide()
    """

    def __init__(self) -> None:
        self._panel = None
        self._title_label = None
        self._detail_label = None
        self._tips_label = None
        self._progress_view = None
        self._lock = threading.Lock()
        self._tips_timer = None
        self._tips_index = 0

    def _ensure_panel(self) -> bool:
        """Create the panel lazily. Must be called from main thread."""
        if self._panel is not None:
            return True

        if not _ensure_appkit():
            return False

        try:
            import AppKit  # noqa: F401

            screen = _get_active_screen()
            if screen is None:
                return False
            screen_frame = screen.frame()

            x = screen_frame.origin.x + (screen_frame.size.width - _DL_WIDTH) / 2
            y = screen_frame.origin.y + (screen_frame.size.height - _DL_HEIGHT) / 2
            frame = ((x, y), (_DL_WIDTH, _DL_HEIGHT))

            panel = _NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                _BORDERLESS | _NON_ACTIVATING,
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(25)  # NSStatusWindowLevel
            panel.setOpaque_(False)
            panel.setBackgroundColor_(_NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces
            panel.setAlphaValue_(0.0)

            # Background view
            BgView = _DownloadPanelView.get_class()
            bg_view = BgView.alloc().initWithFrame_(((0, 0), (_DL_WIDTH, _DL_HEIGHT)))
            panel.contentView().addSubview_(bg_view)

            padding = 30

            # Title label: "Setting up Veery..."
            title_label = _NSTextField.alloc().initWithFrame_(
                ((padding, _DL_HEIGHT - 50), (_DL_WIDTH - 2 * padding, 24))
            )
            title_label.setStringValue_("Setting up Veery...")
            title_label.setBezeled_(False)
            title_label.setDrawsBackground_(False)
            title_label.setEditable_(False)
            title_label.setSelectable_(False)
            title_label.setAlignment_(0)  # NSTextAlignmentLeft
            title_label.setTextColor_(_NSColor.whiteColor())
            title_label.setFont_(_NSFont.systemFontOfSize_weight_(16, 0.23))
            panel.contentView().addSubview_(title_label)

            # Detail label
            detail_label = _NSTextField.alloc().initWithFrame_(
                ((padding, _DL_HEIGHT - 80), (_DL_WIDTH - 2 * padding, 18))
            )
            detail_label.setStringValue_("Preparing...")
            detail_label.setBezeled_(False)
            detail_label.setDrawsBackground_(False)
            detail_label.setEditable_(False)
            detail_label.setSelectable_(False)
            detail_label.setAlignment_(0)
            detail_label.setTextColor_(_NSColor.whiteColor())
            detail_label.setFont_(_NSFont.systemFontOfSize_(13))
            panel.contentView().addSubview_(detail_label)

            # Progress bar
            bar_width = _DL_WIDTH - 2 * padding
            PBarView = _ProgressBarView.get_class()
            progress_view = PBarView.alloc().initWithFrame_(
                ((padding, _DL_HEIGHT - 105), (bar_width, 12))
            )
            panel.contentView().addSubview_(progress_view)

            # Tips label
            tips_label = _NSTextField.alloc().initWithFrame_(
                ((padding, 20), (_DL_WIDTH - 2 * padding, 16))
            )
            tips_label.setStringValue_(_TIPS[0])
            tips_label.setBezeled_(False)
            tips_label.setDrawsBackground_(False)
            tips_label.setEditable_(False)
            tips_label.setSelectable_(False)
            tips_label.setAlignment_(0)
            tips_label.setTextColor_(
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, _TIPS_ALPHA)
            )
            tips_label.setFont_(_NSFont.systemFontOfSize_(12))
            panel.contentView().addSubview_(tips_label)

            self._panel = panel
            self._title_label = title_label
            self._detail_label = detail_label
            self._tips_label = tips_label
            self._progress_view = progress_view
            return True

        except Exception:
            logger.exception("Failed to create download progress panel")
            return False

    def _stop_timers(self) -> None:
        """Invalidate tips cycling timer. Must be called from main thread."""
        if self._tips_timer is not None:
            self._tips_timer.invalidate()
            self._tips_timer = None

    def _reposition_panel(self) -> None:
        """Update panel position to follow the active screen. Must be called from main thread."""
        if self._panel is None:
            return
        screen = _get_active_screen()
        if screen is None:
            return
        screen_frame = screen.frame()
        x = screen_frame.origin.x + (screen_frame.size.width - _DL_WIDTH) / 2
        y = screen_frame.origin.y + (screen_frame.size.height - _DL_HEIGHT) / 2
        self._panel.setFrameOrigin_((x, y))

    def _cycle_tips(self) -> None:
        """Advance to the next tip. Called from main thread by NSTimer."""
        self._tips_index = (self._tips_index + 1) % len(_TIPS)
        if self._tips_label is not None:
            self._tips_label.setStringValue_(_TIPS[self._tips_index])

    def show(self) -> None:
        """Show the download progress overlay."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._stop_timers()
                self._reposition_panel()
                self._tips_index = 0
                if self._tips_label is not None:
                    self._tips_label.setStringValue_(_TIPS[0])
                if self._progress_view is not None:
                    self._progress_view._fraction = 0.0
                    self._progress_view.setNeedsDisplay_(True)
                self._panel.orderFrontRegardless()

                # Fade in
                _NSAnimationContext.beginGrouping()
                _NSAnimationContext.currentContext().setDuration_(0.2)
                self._panel.animator().setAlphaValue_(1.0)
                _NSAnimationContext.endGrouping()

                # Tips cycling timer (every 5 seconds)
                self._tips_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        5.0, self._progress_view, b"cycleTipsTimer:", None, True
                    )
                )
                # Store a reference so the timer callback can find us
                self._progress_view._overlay = self

        # Patch the timer callback to call our _cycle_tips
        def _patch_timer_callback():
            """Override the ObjC cycleTipsTimer_ to route to our Python method."""
            PBarClass = _ProgressBarView.get_class()

            def patched_cycleTipsTimer_(view_self, timer):
                overlay = getattr(view_self, "_overlay", None)
                if overlay is not None:
                    overlay._cycle_tips()

            PBarClass.cycleTipsTimer_ = patched_cycleTipsTimer_

        try:
            _patch_timer_callback()
        except Exception:
            logger.debug("Could not patch tips timer callback")

        _run_on_main(_show)

    def set_progress(self, fraction: float, detail: str) -> None:
        """Update the progress bar and detail text.

        Args:
            fraction: Progress from 0.0 to 1.0.
            detail: Text shown below the title (e.g., "Downloading SenseVoice-Small (1.2 GB)...").
        """
        clamped = max(0.0, min(fraction, 1.0))

        def _update():
            with self._lock:
                if self._progress_view is not None:
                    self._progress_view._fraction = clamped
                    self._progress_view.setNeedsDisplay_(True)
                if self._detail_label is not None:
                    self._detail_label.setStringValue_(detail)

        _run_on_main(_update)

    def hide(self) -> None:
        """Hide the download progress overlay."""
        def _hide():
            with self._lock:
                self._stop_timers()
                if self._panel is not None:
                    _NSAnimationContext.beginGrouping()
                    _NSAnimationContext.currentContext().setDuration_(0.2)
                    self._panel.animator().setAlphaValue_(0.0)
                    _NSAnimationContext.endGrouping()

        _run_on_main(_hide)


# ---- Permission guide overlay ----

# Permission guide dimensions
_PERM_WIDTH = 420
_PERM_HEIGHT = 200
_PERM_RADIUS = 20
_PERM_POLL_INTERVAL = 2.0  # seconds between permission checks

# System Settings deep links
_ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
_MICROPHONE_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"

_PERM_STEPS = (
    {
        "title": "Veery needs your permission",
        "body": (
            "Veery needs Accessibility access to type dictated\n"
            "text into your apps.\n"
            "\n"
            "System Settings \u2192 Privacy & Security \u2192 Accessibility"
        ),
        "step_label": "Step 1 of 2",
        "settings_url": _ACCESSIBILITY_URL,
    },
    {
        "title": "Veery needs your permission",
        "body": (
            "Veery needs Microphone access to hear your speech.\n"
            "\n"
            "System Settings \u2192 Privacy & Security \u2192 Microphone"
        ),
        "step_label": "Step 2 of 2",
        "settings_url": _MICROPHONE_URL,
    },
)


def _check_accessibility() -> bool:
    """Check if Accessibility (AXIsProcessTrusted) is granted."""
    lib_path = ctypes.util.find_library("ApplicationServices")
    if lib_path is None:
        return False
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        logger.debug("Could not check AXIsProcessTrusted")
        return False


def _check_microphone() -> bool:
    """Check if Microphone permission is granted via AVFoundation.

    Returns True if AVFoundation is not available (e.g. missing PyObjC framework),
    since macOS will prompt for microphone access automatically when audio is opened.
    """
    try:
        import AVFoundation

        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeAudio
        )
        # 3 = AVAuthorizationStatusAuthorized
        return status == 3
    except ImportError:
        logger.debug("AVFoundation not available, skipping microphone check")
        return True
    except Exception:
        logger.debug("Could not check microphone authorization")
        return True


def _request_microphone() -> None:
    """Trigger the macOS microphone permission prompt."""
    try:
        import AVFoundation

        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeAudio, lambda granted: None
        )
    except Exception:
        logger.debug("Could not request microphone access")


def check_permissions_granted() -> bool:
    """Return True if both Accessibility and Microphone permissions are granted."""
    return _check_accessibility() and _check_microphone()


class _PermissionPanelView:
    """Custom NSView subclass that draws the permission guide panel background."""

    _ViewClass = None

    @classmethod
    def get_class(cls):
        if cls._ViewClass is not None:
            return cls._ViewClass

        import objc  # noqa: F401

        NSView = _NSView

        class PermissionBackgroundView(NSView):
            """Rounded-rect background for the permission guide."""

            def initWithFrame_(self, frame):
                self = objc.super(PermissionBackgroundView, self).initWithFrame_(frame)
                return self

            def drawRect_(self, rect):
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    _BG_RED, _BG_GREEN, _BG_BLUE, _BG_ALPHA
                ).set()
                path = _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, _PERM_RADIUS, _PERM_RADIUS
                )
                path.fill()

            def pollPermissionTimer_(self, timer):
                """NSTimer callback for permission polling (selector target)."""
                pass  # handled by PermissionGuideOverlay

        cls._ViewClass = PermissionBackgroundView
        return cls._ViewClass


class PermissionGuideOverlay:
    """Overlay that guides users through granting macOS permissions on first launch.

    Usage:
        overlay = PermissionGuideOverlay()
        overlay.show(on_complete=lambda: print("All permissions granted!"))
        overlay.hide()
    """

    def __init__(self) -> None:
        self._panel = None
        self._title_label = None
        self._body_label = None
        self._step_label = None
        self._bg_view = None
        self._lock = threading.Lock()
        self._poll_timer = None
        self._on_complete: Callable[[], None] | None = None
        self._current_step = 0
        # Which steps actually need to be shown (indices into _PERM_STEPS)
        self._pending_steps: list[int] = []

    def _ensure_panel(self) -> bool:
        """Create the panel lazily. Must be called from main thread."""
        if self._panel is not None:
            return True

        if not _ensure_appkit():
            return False

        try:
            import AppKit  # noqa: F401

            screen = _get_active_screen()
            if screen is None:
                return False
            screen_frame = screen.frame()

            x = screen_frame.origin.x + (screen_frame.size.width - _PERM_WIDTH) / 2
            y = screen_frame.origin.y + (screen_frame.size.height - _PERM_HEIGHT) / 2
            frame = ((x, y), (_PERM_WIDTH, _PERM_HEIGHT))

            panel = _NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                _BORDERLESS | _NON_ACTIVATING,
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(25)  # NSStatusWindowLevel
            panel.setOpaque_(False)
            panel.setBackgroundColor_(_NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces
            panel.setAlphaValue_(0.0)

            # Background view
            BgView = _PermissionPanelView.get_class()
            bg_view = BgView.alloc().initWithFrame_(((0, 0), (_PERM_WIDTH, _PERM_HEIGHT)))
            panel.contentView().addSubview_(bg_view)

            padding = 30

            # Step indicator: "Step 1 of 2" — top-right area
            step_label = _NSTextField.alloc().initWithFrame_(
                ((padding, _PERM_HEIGHT - 42), (_PERM_WIDTH - 2 * padding, 16))
            )
            step_label.setStringValue_("")
            step_label.setBezeled_(False)
            step_label.setDrawsBackground_(False)
            step_label.setEditable_(False)
            step_label.setSelectable_(False)
            step_label.setAlignment_(0)  # NSTextAlignmentLeft
            step_label.setTextColor_(
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.5)
            )
            step_label.setFont_(_NSFont.systemFontOfSize_(12))
            panel.contentView().addSubview_(step_label)

            # Title label
            title_label = _NSTextField.alloc().initWithFrame_(
                ((padding, _PERM_HEIGHT - 70), (_PERM_WIDTH - 2 * padding, 24))
            )
            title_label.setStringValue_("")
            title_label.setBezeled_(False)
            title_label.setDrawsBackground_(False)
            title_label.setEditable_(False)
            title_label.setSelectable_(False)
            title_label.setAlignment_(0)
            title_label.setTextColor_(_NSColor.whiteColor())
            title_label.setFont_(_NSFont.systemFontOfSize_weight_(16, 0.23))  # Medium
            panel.contentView().addSubview_(title_label)

            # Body label (multi-line)
            body_label = _NSTextField.alloc().initWithFrame_(
                ((padding, 20), (_PERM_WIDTH - 2 * padding, 80))
            )
            body_label.setStringValue_("")
            body_label.setBezeled_(False)
            body_label.setDrawsBackground_(False)
            body_label.setEditable_(False)
            body_label.setSelectable_(False)
            body_label.setAlignment_(0)
            body_label.setTextColor_(_NSColor.whiteColor())
            body_label.setFont_(_NSFont.systemFontOfSize_(13))
            panel.contentView().addSubview_(body_label)

            self._panel = panel
            self._title_label = title_label
            self._body_label = body_label
            self._step_label = step_label
            self._bg_view = bg_view
            return True

        except Exception:
            logger.exception("Failed to create permission guide panel")
            return False

    def _stop_timer(self) -> None:
        """Stop the permission polling timer. Must be called from main thread."""
        if self._poll_timer is not None:
            self._poll_timer.invalidate()
            self._poll_timer = None

    def _reposition_panel(self) -> None:
        """Update panel position to follow the active screen. Must be called from main thread."""
        if self._panel is None:
            return
        screen = _get_active_screen()
        if screen is None:
            return
        screen_frame = screen.frame()
        x = screen_frame.origin.x + (screen_frame.size.width - _PERM_WIDTH) / 2
        y = screen_frame.origin.y + (screen_frame.size.height - _PERM_HEIGHT) / 2
        self._panel.setFrameOrigin_((x, y))

    def _update_step_content(self) -> None:
        """Update panel text for the current step. Must be called from main thread."""
        if not self._pending_steps:
            return
        step_index = self._pending_steps[self._current_step]
        step = _PERM_STEPS[step_index]

        if self._step_label is not None:
            total = len(self._pending_steps)
            self._step_label.setStringValue_(f"Step {self._current_step + 1} of {total}")
        if self._title_label is not None:
            self._title_label.setStringValue_(step["title"])
        if self._body_label is not None:
            self._body_label.setStringValue_(step["body"])

    def _open_settings(self, url: str) -> None:
        """Open System Settings to a specific pane."""
        subprocess.run(["open", url], check=False)

    def _check_current_permission(self) -> bool:
        """Check if the permission for the current step is granted."""
        if not self._pending_steps:
            return True
        step_index = self._pending_steps[self._current_step]
        if step_index == 0:
            return _check_accessibility()
        return _check_microphone()

    def _poll_permission(self) -> None:
        """Called every 2s to check if the current permission has been granted."""
        if self._check_current_permission():
            self._current_step += 1
            if self._current_step >= len(self._pending_steps):
                # All permissions granted
                logger.info("All permissions granted")
                self._stop_timer()
                self._fade_out()
                if self._on_complete is not None:
                    self._on_complete()
            else:
                # Advance to next step
                self._update_step_content()
                step_index = self._pending_steps[self._current_step]
                step = _PERM_STEPS[step_index]
                self._open_settings(step["settings_url"])
                if step_index == 1:
                    _request_microphone()

    def _fade_in(self) -> None:
        """Fade the panel in. Must be called from main thread."""
        if self._panel is None:
            return
        _NSAnimationContext.beginGrouping()
        _NSAnimationContext.currentContext().setDuration_(0.2)
        self._panel.animator().setAlphaValue_(1.0)
        _NSAnimationContext.endGrouping()

    def _fade_out(self) -> None:
        """Fade the panel out. Must be called from main thread."""
        if self._panel is None:
            return
        _NSAnimationContext.beginGrouping()
        _NSAnimationContext.currentContext().setDuration_(0.2)
        self._panel.animator().setAlphaValue_(0.0)
        _NSAnimationContext.endGrouping()

    def show(self, on_complete: Callable[[], None]) -> None:
        """Show the permission guide overlay.

        Args:
            on_complete: Called when all permissions have been granted.
        """
        self._on_complete = on_complete

        # Determine which permissions are missing
        self._pending_steps = []
        if not _check_accessibility():
            self._pending_steps.append(0)
        if not _check_microphone():
            self._pending_steps.append(1)

        if not self._pending_steps:
            logger.info("All permissions already granted, skipping guide")
            on_complete()
            return

        self._current_step = 0

        def _show():
            with self._lock:
                if not self._ensure_panel():
                    logger.error("Could not create permission guide panel")
                    on_complete()
                    return

                self._update_step_content()
                self._reposition_panel()
                self._panel.orderFrontRegardless()
                self._fade_in()

                # Open Settings for the first step
                step_index = self._pending_steps[0]
                step = _PERM_STEPS[step_index]
                self._open_settings(step["settings_url"])

                # Trigger microphone prompt if that's the first step
                if step_index == 1:
                    _request_microphone()

                # Patch the timer callback to route to our _poll_permission
                bg_view = self._bg_view
                bg_view._perm_overlay = self

                PanelViewClass = _PermissionPanelView.get_class()

                def patched_poll(view_self, timer):
                    overlay = getattr(view_self, "_perm_overlay", None)
                    if overlay is not None:
                        overlay._poll_permission()

                PanelViewClass.pollPermissionTimer_ = patched_poll

                # Start polling timer
                self._poll_timer = (
                    _NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        _PERM_POLL_INTERVAL, self._bg_view, b"pollPermissionTimer:", None, True
                    )
                )

        _run_on_main(_show)

    def hide(self) -> None:
        """Hide the permission guide overlay."""
        def _hide():
            with self._lock:
                self._stop_timer()
                if self._panel is not None:
                    self._fade_out()

        _run_on_main(_hide)
