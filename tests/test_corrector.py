"""Tests for TextCorrector pipeline (jargon + filler removal)."""

from __future__ import annotations

import pytest

from voiceflow.config import PROJECT_ROOT, JargonConfig
from voiceflow.corrector import TextCorrector, remove_fillers
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
def text_corrector(jargon_corrector: JargonCorrector) -> TextCorrector:
    return TextCorrector(jargon=jargon_corrector)


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
    def test_jargon_exception_falls_back_to_raw(self) -> None:
        """If jargon.correct() raises, corrector falls back to raw text."""
        from unittest.mock import MagicMock

        mock_jargon = MagicMock(spec=JargonCorrector)
        mock_jargon.correct = MagicMock(side_effect=RuntimeError("boom"))
        tc = TextCorrector(jargon=mock_jargon)

        result = tc.correct("some text")
        assert result.raw == "some text"
        assert result.jargon_corrected == "some text"
        assert result.final == "some text"

    def test_punctuation_preserved_in_pipeline(self, text_corrector: TextCorrector) -> None:
        """End-to-end: punctuation survives the full pipeline."""
        result = text_corrector.correct("the sharp ratio, is 2.5.")
        assert "Sharpe ratio," in result.jargon_corrected
        assert result.jargon_corrected.endswith(".")


# ---------------------------------------------------------------------------
# Filler word removal tests
# ---------------------------------------------------------------------------


class TestFillerRemoval:
    def test_english_fillers(self) -> None:
        assert remove_fillers("um I think the API is uh broken") == "I think the API is broken"

    def test_chinese_fillers(self) -> None:
        assert remove_fillers("嗯这个API额有问题") == "这个API有问题"

    def test_no_fillers(self) -> None:
        assert remove_fillers("the API is working") == "the API is working"

    def test_empty_string(self) -> None:
        assert remove_fillers("") == ""

    def test_only_fillers(self) -> None:
        result = remove_fillers("um uh")
        assert result == ""

    def test_preserves_legitimate_words(self) -> None:
        # "like" as a verb should not be removed (word-boundary aware)
        assert "like" in remove_fillers("I like this feature")

    def test_mixed_fillers(self) -> None:
        result = remove_fillers("um 嗯 the Kubernetes API you know is 额 broken")
        assert "um" not in result.lower()
        assert "嗯" not in result
        assert "额" not in result
        assert "Kubernetes" in result
        assert "API" in result

    def test_filler_in_pipeline(self, text_corrector: TextCorrector) -> None:
        """Fillers are removed in the full correction pipeline."""
        result = text_corrector.correct("um the sharp ratio is uh 2.5")
        assert "um" not in result.final.lower()
        assert "uh" not in result.final.lower()
        assert "Sharpe ratio" in result.final
