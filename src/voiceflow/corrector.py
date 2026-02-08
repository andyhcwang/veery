"""Text correction pipeline: jargon replacement and filler removal."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from voiceflow.jargon import JargonCorrector

logger = logging.getLogger(__name__)

# Filler words/phrases to remove (case-insensitive, word-boundary-aware).
# Combined into a single alternation regex for one-pass removal instead
# of iterating 12 separate patterns sequentially.
_EN_FILLERS = [
    r"\bum\b", r"\buh\b", r"\bumm\b", r"\buhh\b",
    r"\byou know\b", r"\bI mean\b",
    r"\bbasically\b", r"\bliterally\b",
]
_ZH_FILLERS = [
    "嗯", "额", "啊", "呃",
    "那个", "就是说", "然后吧",
]
_FILLER_PATTERN = re.compile(
    "|".join(_EN_FILLERS + [re.escape(f) for f in _ZH_FILLERS]),
    re.IGNORECASE,
)
# Clean up multiple spaces left after removal
_MULTI_SPACE = re.compile(r"  +")


def remove_fillers(text: str) -> str:
    """Remove common filler words/sounds from transcribed text."""
    text = _FILLER_PATTERN.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


@dataclass
class CorrectionResult:
    """Tracks text through each correction stage."""

    raw: str  # Original STT output
    jargon_corrected: str  # After jargon dict correction
    final: str  # After filler removal


class TextCorrector:
    """Orchestrates the correction pipeline: jargon -> fillers."""

    def __init__(self, jargon: JargonCorrector) -> None:
        self._jargon = jargon

    def correct(self, raw_text: str) -> CorrectionResult:
        """Run the full correction pipeline on raw STT output."""
        if not raw_text:
            return CorrectionResult(raw="", jargon_corrected="", final="")

        # Stage 1: Jargon correction
        try:
            jargon_corrected = self._jargon.correct(raw_text)
        except Exception:
            logger.exception("Jargon correction failed, using raw text")
            jargon_corrected = raw_text
        logger.debug("Jargon stage: %r -> %r", raw_text, jargon_corrected)

        # Stage 2: Filler word removal
        final = remove_fillers(jargon_corrected)
        if final != jargon_corrected:
            logger.debug("Filler removal: %r -> %r", jargon_corrected, final)

        return CorrectionResult(
            raw=raw_text,
            jargon_corrected=jargon_corrected,
            final=final,
        )
