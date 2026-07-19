"""Text correction pipeline: jargon replacement and filler removal."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from veery.jargon import JargonCorrector

logger = logging.getLogger(__name__)

# Filler words/phrases to remove (case-insensitive, word-boundary-aware).
# Only unambiguous fillers are removed unconditionally. Words that carry real
# meaning ("basically", "you know", 那个, 啊, 就是说) must never be blanket-
# stripped — dictation must not delete words the user actually said.
_EN_FILLERS = [
    r"\bum\b", r"\buh\b", r"\bumm\b", r"\buhh\b",
]
# 嗯 never forms real words; 呃 essentially never does in dictation (呃逆 is
# rare medical vocabulary we accept losing). 额 does form real words (金额,
# 额度, 额外), so it is only a filler when not adjacent to another CJK char.
_CJK_RANGE = "一-鿿"
_ZH_FILLERS = [
    "嗯", "呃",
    f"(?<![{_CJK_RANGE}])额(?![{_CJK_RANGE}])",
]
_FILLER_PATTERN = re.compile(
    "|".join(_EN_FILLERS + _ZH_FILLERS),
    re.IGNORECASE,
)
# Stuttered 那个那个(那个)... is filler; a single 那个 is a real demonstrative.
_NAGE_STUTTER = re.compile(r"(?:那个\s*){2,}")
# Clean up multiple spaces left after removal
_MULTI_SPACE = re.compile(r"  +")

# Whisper picks Traditional or Simplified Han per utterance almost at random
# for Mandarin speech; normalize to one script so output is consistent and
# jargon matching sees the same characters the dictionaries use.
_ZH_LOCALE = {"simplified": "zh-cn", "traditional": "zh-tw"}


def normalize_han(text: str, variant: str) -> str:
    """Convert Han characters to one script variant ("off" passes through)."""
    locale = _ZH_LOCALE.get(variant)
    if not locale or not text:
        return text
    try:
        from zhconv import convert

        return convert(text, locale)
    except Exception:
        logger.exception("Han script normalization failed, using original text")
        return text


def remove_fillers(text: str) -> str:
    """Remove common filler words/sounds from transcribed text."""
    text = _NAGE_STUTTER.sub("那个", text)
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
    """Orchestrates the correction pipeline: han script -> jargon -> fillers."""

    def __init__(self, jargon: JargonCorrector, chinese_variant: str = "simplified") -> None:
        self._jargon = jargon
        self._chinese_variant = chinese_variant

    @property
    def jargon(self) -> JargonCorrector:
        return self._jargon

    def correct(self, raw_text: str) -> CorrectionResult:
        """Run the full correction pipeline on raw STT output."""
        if not raw_text:
            return CorrectionResult(raw="", jargon_corrected="", final="")

        # Stage 0: Han script normalization (before jargon so fuzzy matching
        # sees the same script the dictionaries are written in)
        normalized = normalize_han(raw_text, self._chinese_variant)
        if normalized != raw_text:
            logger.debug("Han normalization: %r -> %r", raw_text, normalized)

        # Stage 1: Jargon correction
        try:
            jargon_corrected = self._jargon.correct(normalized)
        except Exception:
            logger.exception("Jargon correction failed, using raw text")
            jargon_corrected = normalized
        logger.debug("Jargon stage: %r -> %r", normalized, jargon_corrected)

        # Stage 2: Filler word removal
        final = remove_fillers(jargon_corrected)
        if final != jargon_corrected:
            logger.debug("Filler removal: %r -> %r", jargon_corrected, final)

        return CorrectionResult(
            raw=raw_text,
            jargon_corrected=jargon_corrected,
            final=final,
        )
