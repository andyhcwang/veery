"""Tests for text output: CGEvent typing, clipboard paste, method selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from veery.config import OutputConfig

# ---------------------------------------------------------------------------
# paste_to_active_app — method selection
# ---------------------------------------------------------------------------


class TestPasteMethodSelection:
    def test_empty_text_returns_immediately(self) -> None:
        with (
            patch("veery.output._type_via_cgevent") as mock_type,
            patch("veery.output._paste_via_clipboard") as mock_paste,
        ):
            from veery.output import paste_to_active_app
            paste_to_active_app("")
        mock_type.assert_not_called()
        mock_paste.assert_not_called()

    def test_short_text_uses_cgevent(self) -> None:
        cfg = OutputConfig(cgevent_char_limit=500)
        with (
            patch("veery.output._type_via_cgevent") as mock_type,
            patch("veery.output._paste_via_clipboard") as mock_paste,
        ):
            from veery.output import paste_to_active_app
            paste_to_active_app("hello", cfg)
        mock_type.assert_called_once_with("hello")
        mock_paste.assert_not_called()

    def test_long_text_uses_clipboard(self) -> None:
        cfg = OutputConfig(cgevent_char_limit=5)
        with (
            patch("veery.output._type_via_cgevent") as mock_type,
            patch("veery.output._paste_via_clipboard") as mock_paste,
        ):
            from veery.output import paste_to_active_app
            paste_to_active_app("hello world", cfg)
        mock_type.assert_not_called()
        mock_paste.assert_called_once()

    def test_boundary_exactly_at_limit_uses_cgevent(self) -> None:
        """Text length == limit should use CGEvent (<=)."""
        cfg = OutputConfig(cgevent_char_limit=5)
        with (
            patch("veery.output._type_via_cgevent") as mock_type,
            patch("veery.output._paste_via_clipboard") as mock_paste,
        ):
            from veery.output import paste_to_active_app
            paste_to_active_app("abcde", cfg)  # exactly 5 chars
        mock_type.assert_called_once()
        mock_paste.assert_not_called()

    def test_default_config_used_when_none(self) -> None:
        with (
            patch("veery.output._type_via_cgevent") as mock_type,
            patch("veery.output._paste_via_clipboard"),
        ):
            from veery.output import paste_to_active_app
            paste_to_active_app("hi")  # no config passed
        mock_type.assert_called_once_with("hi")

    def test_exception_propagates(self) -> None:
        """paste_to_active_app lets exceptions propagate to the caller."""
        import pytest

        with patch("veery.output._type_via_cgevent", side_effect=RuntimeError("CGEvent fail")):
            from veery.output import paste_to_active_app
            with pytest.raises(RuntimeError, match="CGEvent fail"):
                paste_to_active_app("hello")


# ---------------------------------------------------------------------------
# _type_via_cgevent
# ---------------------------------------------------------------------------


class TestTypeViaCGEvent:
    def test_single_batch(self) -> None:
        mock_quartz = MagicMock()
        with patch.dict("sys.modules", {"Quartz": mock_quartz}):
            from veery.output import _type_via_cgevent
            _type_via_cgevent("hello")

        # One batch → 2 CGEventPost calls (key down + key up)
        assert mock_quartz.CGEventPost.call_count == 2

    def test_multiple_batches(self) -> None:
        mock_quartz = MagicMock()
        with patch.dict("sys.modules", {"Quartz": mock_quartz}):
            from veery.output import _type_via_cgevent
            _type_via_cgevent("a" * 45)  # 45 chars → ceil(45/20) = 3 batches

        # 3 batches × 2 events each = 6
        assert mock_quartz.CGEventPost.call_count == 6

    def test_unicode_utf16_length(self) -> None:
        """Emoji requires surrogate pairs in UTF-16 — verify utf16_len calculation."""
        mock_quartz = MagicMock()
        with patch.dict("sys.modules", {"Quartz": mock_quartz}):
            from veery.output import _type_via_cgevent
            # 🎤 is a single Python char but 2 UTF-16 code units
            _type_via_cgevent("\U0001f3a4")

        mock_quartz.CGEventKeyboardSetUnicodeString.assert_called()
        # First call args: (event, utf16_len, text)
        first_call = mock_quartz.CGEventKeyboardSetUnicodeString.call_args_list[0]
        utf16_len = first_call[0][1]
        assert utf16_len == 2  # surrogate pair


# ---------------------------------------------------------------------------
# _paste_via_clipboard
# ---------------------------------------------------------------------------


class TestPasteViaClipboard:
    def test_paste_simulates_cmd_v(self) -> None:
        mock_quartz = MagicMock()
        mock_appkit = MagicMock()
        mock_pb = MagicMock()
        mock_appkit.NSPasteboard.generalPasteboard.return_value = mock_pb
        mock_pb.pasteboardItems.return_value = []

        with (
            patch.dict("sys.modules", {"Quartz": mock_quartz, "AppKit": mock_appkit}),
            patch("veery.output.time.sleep"),
        ):
            from veery.output import _paste_via_clipboard
            _paste_via_clipboard("hello world")

        # Should write text to clipboard
        mock_pb.setString_forType_.assert_called_once()
        # Should simulate Cmd+V (2 CGEventPost calls: down + up)
        assert mock_quartz.CGEventPost.call_count == 2

    def test_paste_restores_clipboard(self) -> None:
        mock_quartz = MagicMock()
        mock_appkit = MagicMock()
        mock_pb = MagicMock()
        mock_appkit.NSPasteboard.generalPasteboard.return_value = mock_pb

        # Simulate existing clipboard content
        mock_item = MagicMock()
        mock_item.types.return_value = ["public.utf8-plain-text"]
        mock_item.dataForType_.return_value = b"original"
        mock_pb.pasteboardItems.return_value = [mock_item]

        with (
            patch.dict("sys.modules", {"Quartz": mock_quartz, "AppKit": mock_appkit}),
            patch("veery.output.time.sleep"),
        ):
            from veery.output import _paste_via_clipboard
            _paste_via_clipboard("new text")

        # clearContents called twice: once before write, once before restore
        assert mock_pb.clearContents.call_count == 2
        # writeObjects_ called to restore
        mock_pb.writeObjects_.assert_called_once()
