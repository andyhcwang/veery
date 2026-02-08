"""Text correction pipeline: jargon replacement then grammar polishing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from voiceflow.grammar import GrammarPolisher
from voiceflow.jargon import JargonCorrector

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Tracks text through each correction stage."""

    raw: str  # Original STT output
    jargon_corrected: str  # After jargon dict correction
    final: str  # After grammar polish (or jargon_corrected if grammar disabled/failed)


class TextCorrector:
    """Orchestrates the correction pipeline: jargon -> grammar."""

    def __init__(self, jargon: JargonCorrector, grammar: GrammarPolisher) -> None:
        self._jargon = jargon
        self._grammar = grammar

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

        # Stage 2: Grammar polishing
        final = self._grammar.polish(jargon_corrected)
        logger.debug("Grammar stage: %r -> %r", jargon_corrected, final)

        return CorrectionResult(
            raw=raw_text,
            jargon_corrected=jargon_corrected,
            final=final,
        )
