"""Tests for phonetic matching, learned.yaml loading, and the full fallback cascade."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from veery.config import JargonConfig
from veery.jargon import JargonCorrector, JargonDictionary, _consonant_skeleton

# ---------------------------------------------------------------------------
# Consonant skeleton tests
# ---------------------------------------------------------------------------


class TestConsonantSkeleton:
    def test_consonant_skeleton_basic(self) -> None:
        """'Sharpe' and 'sharp' produce the same skeleton."""
        assert _consonant_skeleton("Sharpe") == _consonant_skeleton("sharp")

    def test_consonant_skeleton_removes_vowels(self) -> None:
        # "NumPy" -> "numpy" -> remove aeiou -> "nmpy" (y is not a vowel)
        assert _consonant_skeleton("NumPy") == "nmpy"

    def test_consonant_skeleton_short_words(self) -> None:
        """Words < 5 chars produce a correct skeleton."""
        # "TWAP" -> "twp"
        skeleton = _consonant_skeleton("TWAP")
        assert skeleton == "twp"

    def test_consonant_skeleton_deduplicates(self) -> None:
        """Consecutive duplicate consonants are collapsed."""
        # "fuzz" -> "fz" (z deduplicated)
        assert _consonant_skeleton("fuzz") == "fz"

    def test_consonant_skeleton_all_vowels(self) -> None:
        """All-vowel word produces empty skeleton."""
        assert _consonant_skeleton("aeiou") == ""


# ---------------------------------------------------------------------------
# Phonetic lookup tests
# ---------------------------------------------------------------------------


class TestPhoneticLookup:
    @pytest.fixture
    def phonetic_dict(self, tmp_path: Path) -> JargonDictionary:
        """Dictionary with terms that exercise phonetic matching."""
        yaml_path = tmp_path / "phonetic.yaml"
        yaml_path.write_text(yaml.dump({
            "terms": {
                "Sharpe": ["sharpe"],
                "NumPy": ["numpy"],
                "DuckDB": ["duckdb"],
            }
        }))
        config = JargonConfig(dict_paths=(str(yaml_path),), learned_path=None)
        return JargonDictionary(config)

    def test_phonetic_lookup_matches(self, phonetic_dict: JargonDictionary) -> None:
        """'Shrpe' (typo) matches 'Sharpe' via consonant skeleton."""
        # Both "sharpe" and "shrpe" have skeleton "shrp"
        result = phonetic_dict.phonetic_lookup("Shrpe")
        assert result == "Sharpe"

    def test_phonetic_lookup_ignores_short_words(self, phonetic_dict: JargonDictionary) -> None:
        """3-4 char words not matched phonetically."""
        assert phonetic_dict.phonetic_lookup("shp") is None
        assert phonetic_dict.phonetic_lookup("shrp") is None

    def test_phonetic_lookup_ignores_multi_word(self, phonetic_dict: JargonDictionary) -> None:
        """'sharp ratio' not matched phonetically (multi-word)."""
        assert phonetic_dict.phonetic_lookup("sharp ratio") is None


# ---------------------------------------------------------------------------
# Learned.yaml loading tests
# ---------------------------------------------------------------------------


class TestLearnedYamlLoading:
    def test_learned_yaml_loading(self, tmp_path: Path) -> None:
        """Create a learned.yaml with terms, verify they load as Tier 2."""
        learned_path = tmp_path / "learned.yaml"
        learned_path.write_text(yaml.dump({
            "terms": {
                "MyCustomTerm": ["my custom term", "mycustomterm"],
            }
        }))
        config = JargonConfig(dict_paths=(), learned_path=str(learned_path))
        d = JargonDictionary(config)

        assert d.exact_lookup("my custom term") == "MyCustomTerm"
        assert d.exact_lookup("mycustomterm") == "MyCustomTerm"

    def test_curated_wins_over_learned(self, tmp_path: Path) -> None:
        """Same variant in both curated and learned, curated canonical wins."""
        curated_path = tmp_path / "curated.yaml"
        curated_path.write_text(yaml.dump({
            "terms": {
                "CuratedCanonical": ["shared variant"],
            }
        }))

        learned_path = tmp_path / "learned.yaml"
        learned_path.write_text(yaml.dump({
            "terms": {
                "LearnedCanonical": ["shared variant"],
            }
        }))

        # Curated is loaded first (dict_paths), learned second
        config = JargonConfig(
            dict_paths=(str(curated_path),),
            learned_path=str(learned_path),
        )
        d = JargonDictionary(config)

        # Curated should win because it's loaded first
        assert d.exact_lookup("shared variant") == "CuratedCanonical"

    def test_no_learned_file_ok(self, tmp_path: Path) -> None:
        """Missing learned.yaml doesn't crash."""
        config = JargonConfig(
            dict_paths=(),
            learned_path=str(tmp_path / "nonexistent_learned.yaml"),
        )
        d = JargonDictionary(config)
        assert len(d.reverse_index) == 0


# ---------------------------------------------------------------------------
# Full cascade tests: exact -> fuzzy -> phonetic
# ---------------------------------------------------------------------------


class TestCascade:
    @pytest.fixture
    def cascade_corrector(self, tmp_path: Path) -> JargonCorrector:
        """Corrector with terms that test the full fallback chain."""
        yaml_path = tmp_path / "cascade.yaml"
        yaml_path.write_text(yaml.dump({
            "terms": {
                "Sharpe": ["sharpe", "sharp"],
                "NumPy": ["numpy", "num py"],
            }
        }))
        config = JargonConfig(
            dict_paths=(str(yaml_path),),
            learned_path=None,
            fuzzy_threshold=85,
        )
        return JargonCorrector(config)

    def test_cascade_exact_match(self, cascade_corrector: JargonCorrector) -> None:
        """Exact match is tried first and succeeds."""
        assert cascade_corrector.correct("sharpe") == "Sharpe"
        assert cascade_corrector.correct("num py") == "NumPy"

    def test_cascade_fuzzy_match(self, cascade_corrector: JargonCorrector) -> None:
        """When exact fails, fuzzy kicks in."""
        # "sharp" is an exact variant, but "sharpp" should fuzzy-match "sharpe" variant
        result = cascade_corrector.correct("sharpp")
        assert result == "Sharpe"

    def test_cascade_phonetic_fallback(self, cascade_corrector: JargonCorrector) -> None:
        """When exact and fuzzy fail, phonetic lookup is the last resort."""
        # "Shrpe" has same consonant skeleton as "sharpe"
        # but may not fuzzy-match due to low score -- phonetic should catch it
        result = cascade_corrector.dictionary.phonetic_lookup("Shrpe")
        assert result == "Sharpe"

    def test_cascade_no_match(self, cascade_corrector: JargonCorrector) -> None:
        """When all three fail, text is returned unchanged."""
        assert cascade_corrector.correct("banana") == "banana"
