"""Tests for JargonDictionary and JargonCorrector."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from veery.config import PROJECT_ROOT, JargonConfig
from veery.jargon import JargonCorrector, JargonDictionary

QUANT_DICT = str(PROJECT_ROOT / "jargon" / "quant_finance.yaml")
TECH_DICT = str(PROJECT_ROOT / "jargon" / "tech.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def quant_config() -> JargonConfig:
    return JargonConfig(dict_paths=(QUANT_DICT,))


@pytest.fixture
def tech_config() -> JargonConfig:
    return JargonConfig(dict_paths=(TECH_DICT,))


@pytest.fixture
def both_config() -> JargonConfig:
    return JargonConfig(dict_paths=(QUANT_DICT, TECH_DICT))


@pytest.fixture
def quant_dict(quant_config: JargonConfig) -> JargonDictionary:
    return JargonDictionary(quant_config)


@pytest.fixture
def both_dict(both_config: JargonConfig) -> JargonDictionary:
    return JargonDictionary(both_config)


@pytest.fixture
def corrector(both_config: JargonConfig) -> JargonCorrector:
    return JargonCorrector(both_config)


# ---------------------------------------------------------------------------
# JargonDictionary tests
# ---------------------------------------------------------------------------


class TestJargonDictionary:
    def test_load_single_dict(self, quant_dict: JargonDictionary) -> None:
        ri = quant_dict.reverse_index
        assert "sharp ratio" in ri
        assert ri["sharp ratio"] == "Sharpe ratio"
        assert "pnl" in ri
        assert ri["pnl"] == "PnL"
        assert "tee wap" in ri
        assert ri["tee wap"] == "TWAP"

    def test_load_multiple_dicts(self, both_dict: JargonDictionary) -> None:
        ri = both_dict.reverse_index
        # From quant_finance.yaml
        assert "sharp ratio" in ri
        # From tech.yaml
        assert "pie torch" in ri
        assert ri["pie torch"] == "PyTorch"
        assert "pie test" in ri
        assert ri["pie test"] == "pytest"

    def test_missing_dict_file(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = JargonConfig(dict_paths=("/nonexistent/path/fake.yaml",))
        with caplog.at_level(logging.WARNING):
            d = JargonDictionary(cfg)
        assert len(d.reverse_index) == 0
        assert "not found" in caplog.text.lower()

    def test_empty_dict(self, tmp_path: Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")
        cfg = JargonConfig(dict_paths=(str(empty_yaml),))
        d = JargonDictionary(cfg)
        assert len(d.reverse_index) == 0


# ---------------------------------------------------------------------------
# JargonCorrector tests
# ---------------------------------------------------------------------------


class TestJargonCorrector:
    def test_exact_match_single_word(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pnl") == "PnL"

    def test_exact_match_multi_word(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("sharp ratio") == "Sharpe ratio"

    def test_exact_match_three_words(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("p and l") == "PnL"

    def test_fuzzy_match(self, corrector: JargonCorrector) -> None:
        result = corrector.correct("sharpe raito")
        assert result == "Sharpe ratio"

    def test_no_match_passthrough(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("hello world") == "hello world"

    def test_empty_string(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("") == ""

    def test_multiple_corrections(self, corrector: JargonCorrector) -> None:
        result = corrector.correct("the sharp ratio of this tee wap")
        assert "Sharpe ratio" in result
        assert "TWAP" in result

    def test_case_insensitive(self, corrector: JargonCorrector) -> None:
        result = corrector.correct("SHARP RATIO")
        assert result == "Sharpe ratio"

    def test_preserves_non_jargon_words(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("the quick brown fox") == "the quick brown fox"

    def test_no_false_positives(self, corrector: JargonCorrector) -> None:
        """Common short words like 'the', 'is', 'and' must not trigger fuzzy matches."""
        text = "the is and but for"
        assert corrector.correct(text) == text

    def test_tech_terms(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pie torch") == "PyTorch"
        assert corrector.correct("duck DB") == "DuckDB"

    def test_mixed_domains(self, corrector: JargonCorrector) -> None:
        result = corrector.correct("sharp ratio with duck DB")
        assert "Sharpe ratio" in result
        assert "DuckDB" in result


# ---------------------------------------------------------------------------
# Punctuation preservation tests
# ---------------------------------------------------------------------------


class TestPunctuationPreservation:
    """Trailing/leading punctuation must be preserved after jargon correction."""

    def test_trailing_comma(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("sharp ratio, then continue") == "Sharpe ratio, then continue"

    def test_trailing_semicolon(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("sharp ratio; then continue") == "Sharpe ratio; then continue"

    def test_trailing_period(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pnl.") == "PnL."

    def test_trailing_exclamation(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("sharp ratio!") == "Sharpe ratio!"

    def test_parentheses_around_term(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("(pnl)") == "(PnL)"

    def test_multiple_terms_with_commas(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pnl, gmv, and nav") == "PnL, GMV, and NAV"

    def test_quoted_term(self, corrector: JargonCorrector) -> None:
        result = corrector.correct('"sharp ratio" is fine')
        assert "Sharpe ratio" in result

    def test_colon_after_term(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pnl: 1000") == "PnL: 1000"

    def test_question_mark(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("what is the pnl?") == "what is the PnL?"


# ---------------------------------------------------------------------------
# False positive tests
# ---------------------------------------------------------------------------


class TestFalsePositives:
    """Common English words must not fuzzy-match to jargon terms."""

    def test_red_is_not_redis(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("the red is bright") == "the red is bright"

    def test_pass_not_bps(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("pass") == "pass"

    def test_nap_not_nav(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("nap") == "nap"

    def test_ape_not_api(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("ape") == "ape"

    def test_data_not_delta(self, corrector: JargonCorrector) -> None:
        assert corrector.correct("data") == "data"

    def test_common_sentence_untouched(self, corrector: JargonCorrector) -> None:
        text = "I went to the store and bought some food"
        assert corrector.correct(text) == text


# ---------------------------------------------------------------------------
# Unicode / mixed-language tests
# ---------------------------------------------------------------------------


class TestUnicode:
    """Jargon correction with Chinese-English mixed text."""

    def test_chinese_with_spaced_english_jargon(self, corrector: JargonCorrector) -> None:
        """Chinese + space + English jargon + space + Chinese works."""
        result = corrector.correct("这个 sharp ratio 很好")
        assert "Sharpe ratio" in result

    def test_chinese_only_untouched(self, corrector: JargonCorrector) -> None:
        text = "这是一段中文"
        assert corrector.correct(text) == text

    def test_chinese_adjacent_to_english_fuzzy_matches(self, corrector: JargonCorrector) -> None:
        """With fuzzy_threshold=82, CJK chars glued to English may still fuzzy-match.

        STT models typically insert spaces between scripts, so this edge case is rare.
        The jargon term is correctly identified but surrounding CJK context is lost.
        """
        result = corrector.correct("这个sharp ratio很好")
        # Fuzzy match picks up "sharp ratio" even though tokens are "这个sharp" + "ratio很好"
        assert "Sharpe ratio" in result

    def test_chinese_with_spaced_single_term(self, corrector: JargonCorrector) -> None:
        result = corrector.correct("用 pnl 来计算")
        assert "PnL" in result


# ---------------------------------------------------------------------------
# YAML edge case tests
# ---------------------------------------------------------------------------


class TestYamlEdgeCases:
    """YAML loading must handle malformed files gracefully."""

    def test_none_variants(self, tmp_path: Path) -> None:
        """terms: {Foo: null} should not crash."""
        yaml_path = tmp_path / "none_variants.yaml"
        yaml_path.write_text(yaml.dump({"terms": {"Foo": None}}))
        cfg = JargonConfig(dict_paths=(str(yaml_path),))
        d = JargonDictionary(cfg)
        assert len(d.reverse_index) == 0

    def test_none_in_variant_list(self, tmp_path: Path) -> None:
        """terms: {Foo: [foo, null, bar]} should skip None, keep valid entries."""
        yaml_path = tmp_path / "mixed_variants.yaml"
        yaml_path.write_text(yaml.dump({"terms": {"Foo": ["foo", None, "bar"]}}))
        cfg = JargonConfig(dict_paths=(str(yaml_path),))
        d = JargonDictionary(cfg)
        assert "foo" in d.reverse_index
        assert "bar" in d.reverse_index
        assert len(d.reverse_index) == 2

    def test_integer_variant(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Integer variants should be skipped with a warning, not crash."""
        yaml_path = tmp_path / "int_variant.yaml"
        yaml_path.write_text(yaml.dump({"terms": {"K8s": [123, "k8s"]}}))
        cfg = JargonConfig(dict_paths=(str(yaml_path),))
        with caplog.at_level(logging.WARNING):
            d = JargonDictionary(cfg)
        assert "k8s" in d.reverse_index
        assert len(d.reverse_index) == 1
        assert "non-string" in caplog.text.lower()

    def test_missing_terms_key(self, tmp_path: Path) -> None:
        """YAML without 'terms' key should produce empty dictionary."""
        yaml_path = tmp_path / "no_terms.yaml"
        yaml_path.write_text(yaml.dump({"not_terms": {"Foo": ["foo"]}}))
        cfg = JargonConfig(dict_paths=(str(yaml_path),))
        d = JargonDictionary(cfg)
        assert len(d.reverse_index) == 0

    def test_empty_variant_list(self, tmp_path: Path) -> None:
        """terms: {Foo: []} should produce empty dictionary."""
        yaml_path = tmp_path / "empty_list.yaml"
        yaml_path.write_text(yaml.dump({"terms": {"Foo": []}}))
        cfg = JargonConfig(dict_paths=(str(yaml_path),))
        d = JargonDictionary(cfg)
        assert len(d.reverse_index) == 0
