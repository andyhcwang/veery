"""Tests for TextCorrector pipeline with mocked grammar polisher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voiceflow.config import PROJECT_ROOT, JargonConfig
from voiceflow.corrector import TextCorrector
from voiceflow.grammar import GrammarPolisher
from voiceflow.jargon import JargonCorrector

QUANT_DICT = str(PROJECT_ROOT / "jargon" / "quant_finance.yaml")
TECH_DICT = str(PROJECT_ROOT / "jargon" / "tech.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jargon_corrector() -> JargonCorrector:
    cfg = JargonConfig(dict_paths=(QUANT_DICT, TECH_DICT))
    return JargonCorrector(cfg)


@pytest.fixture
def identity_grammar() -> GrammarPolisher:
    """Mock GrammarPolisher that returns input unchanged."""
    mock = MagicMock(spec=GrammarPolisher)
    mock.polish = MagicMock(side_effect=lambda text: text)
    return mock


@pytest.fixture
def text_corrector(jargon_corrector: JargonCorrector, identity_grammar: GrammarPolisher) -> TextCorrector:
    return TextCorrector(jargon=jargon_corrector, grammar=identity_grammar)


# ---------------------------------------------------------------------------
# CorrectionResult tests
# ---------------------------------------------------------------------------


class TestCorrectionResult:
    def test_empty_input(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("")
        assert result.raw == ""
        assert result.jargon_corrected == ""
        assert result.final == ""

    def test_jargon_only_correction(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("sharp ratio")
        assert result.raw == "sharp ratio"
        assert result.jargon_corrected == "Sharpe ratio"
        # Mock grammar is identity, so final == jargon_corrected
        assert result.final == result.jargon_corrected

    def test_no_correction_needed(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("hello world")
        assert result.raw == "hello world"
        assert result.jargon_corrected == "hello world"
        assert result.final == "hello world"

    def test_correction_chain(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("the tee wap signal")
        assert result.raw == "the tee wap signal"
        assert "TWAP" in result.jargon_corrected
        assert result.final == result.jargon_corrected


# ---------------------------------------------------------------------------
# Mock grammar tests
# ---------------------------------------------------------------------------


class TestGrammarMock:
    def test_grammar_polisher_applied(self, jargon_corrector: JargonCorrector) -> None:
        """Use a mock that uppercases input, verify final is uppercased."""
        mock_grammar = MagicMock(spec=GrammarPolisher)
        mock_grammar.polish = MagicMock(side_effect=lambda text: text.upper())
        tc = TextCorrector(jargon=jargon_corrector, grammar=mock_grammar)

        result = tc.correct("sharp ratio")
        assert result.jargon_corrected == "Sharpe ratio"
        assert result.final == "SHARPE RATIO"

    def test_grammar_disabled(self, jargon_corrector: JargonCorrector) -> None:
        """When grammar polisher returns input unchanged, final == jargon_corrected."""
        mock_grammar = MagicMock(spec=GrammarPolisher)
        mock_grammar.polish = MagicMock(side_effect=lambda text: text)
        tc = TextCorrector(jargon=jargon_corrector, grammar=mock_grammar)

        result = tc.correct("the pnl report")
        assert result.final == result.jargon_corrected


# ---------------------------------------------------------------------------
# Integration-style tests (fast, no ML model)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_pipeline_quant(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("the sharp ratio is 2.5 for this tee wap")
        assert "Sharpe ratio" in result.jargon_corrected
        assert "TWAP" in result.jargon_corrected

    def test_full_pipeline_tech(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("run pie test on duck DB")
        assert "pytest" in result.jargon_corrected
        assert "DuckDB" in result.jargon_corrected

    def test_full_pipeline_mixed_domain(self, text_corrector: TextCorrector) -> None:
        result = text_corrector.correct("compute sharp ratio in duck DB with pie torch")
        assert "Sharpe ratio" in result.jargon_corrected
        assert "DuckDB" in result.jargon_corrected
        assert "PyTorch" in result.jargon_corrected


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_jargon_exception_falls_back_to_raw(self, identity_grammar: GrammarPolisher) -> None:
        """If jargon.correct() raises, corrector falls back to raw text."""
        mock_jargon = MagicMock(spec=JargonCorrector)
        mock_jargon.correct = MagicMock(side_effect=RuntimeError("boom"))
        tc = TextCorrector(jargon=mock_jargon, grammar=identity_grammar)

        result = tc.correct("some text")
        assert result.raw == "some text"
        assert result.jargon_corrected == "some text"
        assert result.final == "some text"

    def test_grammar_exception_propagates(self, jargon_corrector: JargonCorrector) -> None:
        """If grammar.polish() raises, the exception propagates (grammar handles its own errors internally)."""
        mock_grammar = MagicMock(spec=GrammarPolisher)
        mock_grammar.polish = MagicMock(side_effect=RuntimeError("model crash"))
        tc = TextCorrector(jargon=jargon_corrector, grammar=mock_grammar)

        with pytest.raises(RuntimeError, match="model crash"):
            tc.correct("sharp ratio")

    def test_punctuation_preserved_in_pipeline(self, text_corrector: TextCorrector) -> None:
        """End-to-end: punctuation survives the full pipeline."""
        result = text_corrector.correct("the sharp ratio, is 2.5.")
        assert "Sharpe ratio," in result.jargon_corrected
        assert result.jargon_corrected.endswith(".")
