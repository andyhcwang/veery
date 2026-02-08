"""Floating overlay indicator for recording/processing state.

Shows a small translucent pill at the top-center of the screen that
doesn't steal focus from the active application.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Lazy-loaded AppKit/Quartz references
_NSPanel = None
_NSColor = None
_NSTextField = None
_NSFont = None
_NSScreen = None
_NSView = None
_NSBezierPath = None
_loaded = False


def _ensure_appkit() -> bool:
    """Lazy-load AppKit classes. Returns True if available."""
    global _NSPanel, _NSColor, _NSTextField, _NSFont, _NSScreen, _NSView, _NSBezierPath, _loaded
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
        _loaded = True
        return True
    except ImportError:
        logger.warning("AppKit not available, overlay disabled")
        return False


# Style masks for a borderless, non-activating panel
_BORDERLESS = 0
_NON_ACTIVATING = 1 << 7  # NSWindowStyleMaskNonactivatingPanel


class _PillView:
    """Custom NSView subclass that draws a rounded-rect pill background."""

    _PillViewClass = None

    @classmethod
    def get_class(cls):
        if cls._PillViewClass is not None:
            return cls._PillViewClass

        import objc

        NSView = _NSView

        class PillBackgroundView(NSView):
            def drawRect_(self, rect):
                _NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.85).set()
                path = _NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 16, 16)
                path.fill()

        cls._PillViewClass = PillBackgroundView
        return cls._PillViewClass


class OverlayIndicator:
    """Floating pill overlay that shows recording/processing status.

    Usage:
        overlay = OverlayIndicator()
        overlay.show_recording()   # Shows "🔴 Recording..."
        overlay.show_processing()  # Shows "⏳ Processing..."
        overlay.hide()             # Hides the overlay
    """

    def __init__(self) -> None:
        self._panel = None
        self._label = None
        self._lock = threading.Lock()

    def _ensure_panel(self) -> bool:
        """Create the panel lazily on first use. Must be called from main thread."""
        if self._panel is not None:
            return True

        if not _ensure_appkit():
            return False

        try:
            import AppKit

            screen = _NSScreen.mainScreen()
            if screen is None:
                return False
            screen_frame = screen.frame()

            # Pill dimensions
            pill_width = 200
            pill_height = 40

            # Position: top-center, 80px below top of screen
            x = (screen_frame.size.width - pill_width) / 2
            y = screen_frame.size.height - pill_height - 80

            frame = ((x, y), (pill_width, pill_height))

            # Create non-activating borderless panel
            panel = _NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                _BORDERLESS | _NON_ACTIVATING,
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(25)  # NSStatusWindowLevel — above everything
            panel.setOpaque_(False)
            panel.setBackgroundColor_(_NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)  # Click-through
            panel.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces

            # Add pill background view
            PillView = _PillView.get_class()
            pill_view = PillView.alloc().initWithFrame_(((0, 0), (pill_width, pill_height)))
            panel.contentView().addSubview_(pill_view)

            # Add text label
            label = _NSTextField.alloc().initWithFrame_(((0, 0), (pill_width, pill_height)))
            label.setStringValue_("")
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setAlignment_(1)  # NSTextAlignmentCenter
            label.setTextColor_(_NSColor.whiteColor())
            label.setFont_(_NSFont.systemFontOfSize_weight_(15, 0.5))  # Medium weight
            panel.contentView().addSubview_(label)

            self._panel = panel
            self._label = label
            return True

        except Exception:
            logger.exception("Failed to create overlay panel")
            return False

    def _run_on_main(self, block) -> None:
        """Dispatch a block to the main thread."""
        try:
            from PyObjCTools.AppHelper import callAfter
            callAfter(block)
        except Exception:
            logger.exception("Failed to dispatch to main thread")

    def show_recording(self) -> None:
        """Show the recording indicator."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._label.setStringValue_("\U0001f534  Recording...")
                self._panel.orderFrontRegardless()

        self._run_on_main(_show)

    def show_processing(self) -> None:
        """Show the processing indicator."""
        def _show():
            with self._lock:
                if not self._ensure_panel():
                    return
                self._label.setStringValue_("\u231b  Processing...")
                self._panel.orderFrontRegardless()

        self._run_on_main(_show)

    def hide(self) -> None:
        """Hide the overlay."""
        def _hide():
            with self._lock:
                if self._panel is not None:
                    self._panel.orderOut_(None)

        self._run_on_main(_hide)
