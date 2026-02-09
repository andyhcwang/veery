"""Auto-learning from user corrections via re-dictation detection."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from veery.config import PROJECT_ROOT, LearningConfig

logger = logging.getLogger(__name__)


class CorrectionLearner:
    """Learns jargon corrections from re-dictation within a time window.

    After Veery pastes text, if the user re-dictates similar text within
    a short window, the learner treats the new text as a correction.
    It tracks (variant -> canonical) pairs. After `promotion_threshold`
    identical corrections, the pair is auto-added to learned.yaml.
    """

    def __init__(self, config: LearningConfig) -> None:
        self._config = config
        learned_path = Path(config.learned_path)
        if not learned_path.is_absolute():
            learned_path = PROJECT_ROOT / learned_path
        self._learned_path = learned_path
        self._promotion_threshold = config.promotion_threshold

        # (variant_lower, canonical) -> {"count": int, "first_seen": str}
        self._pending: dict[tuple[str, str], dict] = {}
        self._load_pending()

    def _load_pending(self) -> None:
        """Load pending corrections from learned.yaml's 'pending' section."""
        if not self._learned_path.exists():
            return
        with open(self._learned_path) as f:
            data = yaml.safe_load(f) or {}
        for entry in data.get("pending", []):
            key = (entry["variant"].lower(), entry["canonical"])
            self._pending[key] = {
                "count": entry.get("count", 1),
                "first_seen": entry.get("first_seen", datetime.now(tz=UTC).strftime("%Y-%m-%d")),
            }

    def log_correction(self, original_full: str, correction_phrase: str) -> str | None:
        """Find what the user corrected and log it.

        Uses fuzzy matching to find the substring in original_full that best
        matches the correction_phrase, then records the (old -> new) pair.

        Args:
            original_full: The full text that Veery pasted.
            correction_phrase: The user's re-dictated correction (short phrase).

        Returns:
            The promoted canonical form if threshold was reached, else None.
        """
        if not original_full or not correction_phrase:
            return None

        correction_clean = correction_phrase.strip()
        if not correction_clean:
            return None

        # Find the best matching substring in the original text
        original_words = original_full.split()
        correction_word_count = len(correction_clean.split())

        best_match: str | None = None
        best_score = 0.0

        # Slide a window of the same word count over the original
        for window_size in range(max(1, correction_word_count - 1), correction_word_count + 2):
            for start in range(len(original_words) - window_size + 1):
                candidate = " ".join(original_words[start : start + window_size])
                score = fuzz.ratio(candidate.lower(), correction_clean.lower())
                if score > best_score:
                    best_score = score
                    best_match = candidate

        if best_match is None or best_score < 50:
            logger.info("No matching substring found for correction '%s'", correction_clean)
            return None

        # If the correction is the same as what we found, nothing to learn
        if best_match.lower() == correction_clean.lower():
            logger.info("Correction matches original, nothing to learn")
            return None

        # Record the correction: old_phrase (STT variant) -> correction (canonical)
        variant = best_match.lower()
        canonical = correction_clean
        key = (variant, canonical)

        entry = self._pending.get(key)
        if entry is None:
            entry = {"count": 0, "first_seen": datetime.now(tz=UTC).strftime("%Y-%m-%d")}
            self._pending[key] = entry
        entry["count"] += 1
        count = entry["count"]
        logger.info(
            "Correction logged: '%s' -> '%s' (count: %d/%d)",
            variant,
            canonical,
            count,
            self._promotion_threshold,
        )

        self._save()

        if count >= self._promotion_threshold:
            self._promote(variant, canonical)
            return canonical

        return None

    def _promote(self, variant: str, canonical: str) -> None:
        """Move a correction from pending to the terms section of learned.yaml."""
        logger.info("Promoting learned term: '%s' -> '%s'", variant, canonical)

        data = self._load_yaml()
        terms = data.setdefault("terms", {})

        # Add variant to canonical's list
        if canonical not in terms:
            terms[canonical] = []
        if variant not in terms[canonical]:
            terms[canonical].append(variant)

        # Remove from pending
        pending = data.get("pending", [])
        data["pending"] = [
            e for e in pending if not (e["variant"].lower() == variant and e["canonical"] == canonical)
        ]

        # Also remove from in-memory pending
        self._pending.pop((variant, canonical), None)

        self._save_yaml(data)

    def _save(self) -> None:
        """Save current pending corrections to learned.yaml."""
        data = self._load_yaml()

        # Rebuild pending section from in-memory state
        pending_list = []
        for (variant, canonical), entry in self._pending.items():
            pending_list.append({
                "variant": variant,
                "canonical": canonical,
                "count": entry["count"],
                "first_seen": entry["first_seen"],
            })
        data["pending"] = pending_list

        self._save_yaml(data)

    def _load_yaml(self) -> dict:
        """Load learned.yaml or return empty structure."""
        if self._learned_path.exists():
            with open(self._learned_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_yaml(self, data: dict) -> None:
        """Write data to learned.yaml, creating parent dirs if needed."""
        self._learned_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._learned_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
