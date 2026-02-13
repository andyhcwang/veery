"""Tests for CorrectionLearner auto-learning from user corrections."""

from __future__ import annotations

from pathlib import Path

import yaml

from veery.config import LearningConfig
from veery.learner import CorrectionLearner


def _make_config(tmp_path: Path, **overrides) -> LearningConfig:
    """Create a LearningConfig pointing at tmp_path for all file I/O."""
    defaults = {
        "learned_path": str(tmp_path / "learned.yaml"),
        "promotion_threshold": 3,
    }
    defaults.update(overrides)
    return LearningConfig(**defaults)


class TestLogCorrectionBasic:
    def test_log_correction_basic(self, tmp_path: Path) -> None:
        """Log a simple correction, verify pending count incremented."""
        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)

        result = learner.log_correction("the sharp ratio is good", "Sharpe ratio")
        # Not promoted yet (threshold=3, count=1)
        assert result is None

        # Verify pending was persisted
        data = yaml.safe_load((tmp_path / "learned.yaml").read_text())
        pending = data["pending"]
        assert len(pending) == 1
        assert pending[0]["variant"] == "sharp ratio"
        assert pending[0]["canonical"] == "Sharpe ratio"
        assert pending[0]["count"] == 1


class TestPromotionThreshold:
    def test_promotion_threshold(self, tmp_path: Path) -> None:
        """Log N corrections (N = threshold), verify term appears in learned.yaml terms section."""
        config = _make_config(tmp_path, promotion_threshold=3)
        learner = CorrectionLearner(config)

        # Log 2 corrections -- not promoted yet
        assert learner.log_correction("the sharp ratio", "Sharpe ratio") is None
        assert learner.log_correction("the sharp ratio", "Sharpe ratio") is None

        # 3rd correction triggers promotion
        result = learner.log_correction("the sharp ratio", "Sharpe ratio")
        assert result == "Sharpe ratio"

        # Verify terms section in learned.yaml
        data = yaml.safe_load((tmp_path / "learned.yaml").read_text())
        assert "Sharpe ratio" in data["terms"]
        assert "sharp ratio" in data["terms"]["Sharpe ratio"]

        # Pending should no longer contain this entry
        pending = data.get("pending", [])
        promoted_entries = [e for e in pending if e["canonical"] == "Sharpe ratio"]
        assert len(promoted_entries) == 0


class TestLearnedYamlCreatedFromScratch:
    def test_learned_yaml_created_from_scratch(self, tmp_path: Path) -> None:
        """Start with no learned.yaml, log a correction, verify file is created."""
        learned_path = tmp_path / "subdir" / "learned.yaml"
        assert not learned_path.exists()

        config = LearningConfig(learned_path=str(learned_path))
        learner = CorrectionLearner(config)
        learner.log_correction("the sharp ratio is up", "Sharpe ratio")

        assert learned_path.exists()
        data = yaml.safe_load(learned_path.read_text())
        assert len(data["pending"]) >= 1


class TestPendingPersistence:
    def test_pending_persistence(self, tmp_path: Path) -> None:
        """Log correction, create new learner instance, verify pending count persists."""
        config = _make_config(tmp_path, promotion_threshold=5)

        learner1 = CorrectionLearner(config)
        learner1.log_correction("the sharp ratio is ok", "Sharpe ratio")
        learner1.log_correction("the sharp ratio is ok", "Sharpe ratio")

        # New instance should load persisted pending
        learner2 = CorrectionLearner(config)
        # Third correction on new instance
        learner2.log_correction("the sharp ratio is ok", "Sharpe ratio")

        data = yaml.safe_load((tmp_path / "learned.yaml").read_text())
        entry = [e for e in data["pending"] if e["canonical"] == "Sharpe ratio"][0]
        assert entry["count"] == 3


class TestDuplicateCounting:
    def test_duplicate_counting(self, tmp_path: Path) -> None:
        """Same correction logged multiple times increments the same counter."""
        config = _make_config(tmp_path, promotion_threshold=10)
        learner = CorrectionLearner(config)

        for _ in range(5):
            learner.log_correction("use tee wap orders", "TWAP")

        data = yaml.safe_load((tmp_path / "learned.yaml").read_text())
        twap_entries = [e for e in data["pending"] if e["canonical"] == "TWAP"]
        assert len(twap_entries) == 1
        assert twap_entries[0]["count"] == 5


class TestNoCorrectionWhenSameText:
    def test_no_correction_when_same_text(self, tmp_path: Path) -> None:
        """Correction phrase matches original, nothing learned."""
        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)

        result = learner.log_correction("Sharpe ratio", "Sharpe ratio")
        assert result is None

        # No file should be created (or pending should be empty)
        if (tmp_path / "learned.yaml").exists():
            data = yaml.safe_load((tmp_path / "learned.yaml").read_text()) or {}
            assert len(data.get("pending", [])) == 0


class TestNoCorrectionWhenEmpty:
    def test_no_correction_when_empty_original(self, tmp_path: Path) -> None:
        """Empty original text returns None."""
        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)
        assert learner.log_correction("", "Sharpe ratio") is None

    def test_no_correction_when_empty_correction(self, tmp_path: Path) -> None:
        """Empty correction phrase returns None."""
        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)
        assert learner.log_correction("the sharp ratio", "") is None

    def test_no_correction_when_whitespace_correction(self, tmp_path: Path) -> None:
        """Whitespace-only correction phrase returns None."""
        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)
        assert learner.log_correction("the sharp ratio", "   ") is None


class TestLoadPendingMalformedYaml:
    def test_non_dict_yaml_root(self, tmp_path: Path) -> None:
        """If learned.yaml root is not a dict (e.g., a list), learner loads with empty pending."""
        learned_path = tmp_path / "learned.yaml"
        learned_path.write_text("- item1\n- item2\n")

        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)
        assert learner._pending == {}

    def test_missing_keys_in_pending_entry(self, tmp_path: Path) -> None:
        """Pending entries missing 'variant' or 'canonical' are skipped without crashing."""
        learned_path = tmp_path / "learned.yaml"
        data = {
            "pending": [
                {"variant": "sharp ratio", "canonical": "Sharpe ratio", "count": 2},
                {"variant": "missing canonical"},
                {"canonical": "missing variant"},
                {},
                {"variant": "tee wap", "canonical": "TWAP", "count": 1},
            ]
        }
        learned_path.write_text(yaml.dump(data))

        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)

        # Only the two well-formed entries should be loaded
        assert len(learner._pending) == 2
        assert ("sharp ratio", "Sharpe ratio") in learner._pending
        assert ("tee wap", "TWAP") in learner._pending

    def test_non_dict_pending_entry(self, tmp_path: Path) -> None:
        """A pending entry that is not a dict (e.g., a string) is skipped."""
        learned_path = tmp_path / "learned.yaml"
        data = {
            "pending": [
                "just a string",
                {"variant": "sharp ratio", "canonical": "Sharpe ratio", "count": 1},
            ]
        }
        learned_path.write_text(yaml.dump(data))

        config = _make_config(tmp_path)
        learner = CorrectionLearner(config)

        assert len(learner._pending) == 1
        assert ("sharp ratio", "Sharpe ratio") in learner._pending


class TestMultipleDistinctCorrections:
    def test_multiple_distinct_corrections(self, tmp_path: Path) -> None:
        """Two different corrections tracked independently."""
        config = _make_config(tmp_path, promotion_threshold=10)
        learner = CorrectionLearner(config)

        learner.log_correction("the sharp ratio is high", "Sharpe ratio")
        learner.log_correction("use tee wap orders", "TWAP")
        learner.log_correction("use tee wap orders", "TWAP")

        data = yaml.safe_load((tmp_path / "learned.yaml").read_text())
        pending = data["pending"]

        sharpe_entries = [e for e in pending if e["canonical"] == "Sharpe ratio"]
        twap_entries = [e for e in pending if e["canonical"] == "TWAP"]

        assert len(sharpe_entries) == 1
        assert sharpe_entries[0]["count"] == 1
        assert len(twap_entries) == 1
        assert twap_entries[0]["count"] == 2
