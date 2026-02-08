"""Text output to the active macOS application via CGEvent typing or clipboard paste."""

from __future__ import annotations

import logging
import time

from voiceflow.config import OutputConfig

logger = logging.getLogger(__name__)


def _type_via_cgevent(text: str) -> None:
    """Type text character-by-character via CGEvents. Bypasses clipboard entirely."""
    import Quartz

    for char in text:
        # CGEventKeyboardSetUnicodeString expects UTF-16 length (UniCharCount),
        # not Python str length. Characters outside the BMP (emoji, some CJK)
        # require 2 UTF-16 code units (surrogate pair).
        utf16_len = len(char.encode("utf-16-le")) // 2

        event_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event_down, utf16_len, char)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, event_down)

        event_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(event_up, utf16_len, char)
        Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, event_up)


def _paste_via_clipboard(text: str, paste_delay_sec: float = 0.05) -> None:
    """Save clipboard, paste text via Cmd+V, then restore original clipboard."""
    import AppKit
    import Quartz

    pb = AppKit.NSPasteboard.generalPasteboard()

    # Save ALL pasteboard types (not just plain text)
    old_items: list[dict] = []
    for item in pb.pasteboardItems() or []:
        saved: dict = {}
        for ptype in item.types():
            data = item.dataForType_(ptype)
            if data:
                saved[ptype] = data
        old_items.append(saved)

    # Write our text to the clipboard
    pb.clearContents()
    pb.setString_forType_(text, AppKit.NSPasteboardTypeString)

    # Simulate Cmd+V via CGEvent
    cmd_v_down = Quartz.CGEventCreateKeyboardEvent(None, 9, True)  # 9 = 'v' keycode
    Quartz.CGEventSetFlags(cmd_v_down, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_v_down)

    cmd_v_up = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
    Quartz.CGEventSetFlags(cmd_v_up, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, cmd_v_up)

    # Wait for paste to complete, then restore original clipboard
    time.sleep(paste_delay_sec)

    pb.clearContents()
    items_to_restore = []
    for saved in old_items:
        item = AppKit.NSPasteboardItem.alloc().init()
        for ptype, data in saved.items():
            item.setData_forType_(data, ptype)
        items_to_restore.append(item)
    if items_to_restore:
        pb.writeObjects_(items_to_restore)


def paste_to_active_app(text: str, config: OutputConfig | None = None) -> None:
    """Output text to the active application.

    Short text uses CGEvent character-by-character typing (no clipboard).
    Long text uses clipboard paste with save/restore.
    """
    if not text:
        return

    cfg = config or OutputConfig()

    try:
        if len(text) <= cfg.cgevent_char_limit:
            logger.debug("Typing %d chars via CGEvent", len(text))
            _type_via_cgevent(text)
        else:
            logger.debug("Pasting %d chars via clipboard", len(text))
            _paste_via_clipboard(text, paste_delay_sec=cfg.paste_delay_ms / 1000.0)
    except Exception:
        logger.exception("Failed to output text to active app")
