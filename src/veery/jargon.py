"""Jargon correction via YAML dictionaries and fuzzy matching."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from veery.config import PROJECT_ROOT, JargonConfig

logger = logging.getLogger(__name__)

# Regex to split text into word tokens and non-word separators
_TOKEN_RE = re.compile(r"(\S+)")

# Regex to strip leading/trailing punctuation from a word token, preserving it for re-attachment
_PUNCT_STRIP_RE = re.compile(r"^([^\w]*)(.+?)([^\w]*)$", re.UNICODE)


def _consonant_skeleton(word: str) -> str:
    """Strip vowels and normalize consecutive duplicates.

    Examples: 'Sharpe' -> 'shrp', 'sharp' -> 'shrp', 'NumPy' -> 'nmp'
    """
    word = word.lower()
    skeleton = "".join(c for c in word if c not in "aeiou")
    # Remove consecutive duplicates
    result: list[str] = []
    for c in skeleton:
        if not result or c != result[-1]:
            result.append(c)
    return "".join(result)


class JargonDictionary:
    """Loads YAML jargon files and builds a reverse index (variant -> canonical)."""

    def __init__(self, config: JargonConfig) -> None:
        self._config = config
        # variant (lowercased) -> canonical form
        self._reverse_index: dict[str, str] = {}
        # Variants bucketed by word count for fuzzy matching
        self._variants_by_word_count: dict[int, list[str]] = {}

        # Load curated dictionaries (Tier 1 -- first-loaded wins on conflict)
        for dict_path_str in config.dict_paths:
            dict_path = Path(dict_path_str)
            if not dict_path.is_absolute():
                dict_path = PROJECT_ROOT / dict_path
            self._load_file(dict_path)

        # Load learned dictionary (Tier 2 -- curated takes priority)
        if config.learned_path is not None:
            learned_path = Path(config.learned_path)
            if not learned_path.is_absolute():
                learned_path = PROJECT_ROOT / learned_path
            self._load_file(learned_path)

        # Build word-count buckets for fuzzy matching
        for variant in self._reverse_index:
            wc = len(variant.split())
            self._variants_by_word_count.setdefault(wc, []).append(variant)

        # Build phonetic index (consonant skeleton -> variants)
        self._phonetic_index: dict[str, list[str]] = {}
        for variant in self._reverse_index:
            if " " not in variant and len(variant) >= 5:  # Single words only, >= 5 chars
                skeleton = _consonant_skeleton(variant)
                self._phonetic_index.setdefault(skeleton, []).append(variant)

        logger.info("Loaded %d jargon variants across %d files", len(self._reverse_index), len(config.dict_paths))

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            logger.debug("Jargon file not found (optional): %s", path)
            return

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        terms: dict[str, list[str]] = data.get("terms", {})
        for canonical, variants in terms.items():
            if not variants:
                continue
            for variant in variants:
                if not isinstance(variant, str):
                    logger.warning("Skipping non-string variant %r for '%s'", variant, canonical)
                    continue
                key = variant.lower()
                if key in self._reverse_index and self._reverse_index[key] != canonical:
                    logger.warning(
                        "Jargon conflict: '%s' maps to both '%s' and '%s', keeping first",
                        key,
                        self._reverse_index[key],
                        canonical,
                    )
                    continue
                self._reverse_index[key] = canonical

    def exact_lookup(self, phrase: str) -> str | None:
        """Look up a phrase (case-insensitive). Returns canonical form or None."""
        return self._reverse_index.get(phrase.lower())

    def fuzzy_lookup(self, phrase: str, score_cutoff: int = 85, word_count: int | None = None) -> str | None:
        """Fuzzy match a phrase against variants with the same word count.

        Args:
            phrase: The phrase to look up.
            score_cutoff: Minimum rapidfuzz score (0-100) to accept a match.
            word_count: Only match against variants with this many words.
                        If None, inferred from the phrase.
        """
        if word_count is None:
            word_count = len(phrase.split())

        candidates = self._variants_by_word_count.get(word_count, [])
        if not candidates:
            return None

        result = process.extractOne(phrase.lower(), candidates, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if result is None:
            return None

        matched_variant, _score, _idx = result
        return self._reverse_index[matched_variant]

    def phonetic_lookup(self, word: str) -> str | None:
        """Look up a word by consonant skeleton match. Only for single words >= 5 chars."""
        if " " in word or len(word) < 5:
            return None
        skeleton = _consonant_skeleton(word)
        candidates = self._phonetic_index.get(skeleton, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return self._reverse_index[candidates[0]]
        # Multiple candidates with same skeleton: disambiguate with fuzzy
        result = process.extractOne(word.lower(), candidates, scorer=fuzz.ratio, score_cutoff=70)
        if result:
            return self._reverse_index[result[0]]
        return None

    @property
    def reverse_index(self) -> dict[str, str]:
        return self._reverse_index


class JargonCorrector:
    """Applies jargon correction to transcribed text using sliding-window matching."""

    def __init__(self, config: JargonConfig) -> None:
        self._config = config
        self._dictionary = JargonDictionary(config)

    @property
    def dictionary(self) -> JargonDictionary:
        return self._dictionary

    def correct(self, text: str) -> str:
        """Correct jargon in the given text.

        Uses a sliding window approach: try max_phrase_words-word phrases first,
        then shorter phrases, then single words. Exact match is tried before fuzzy.
        Once a phrase is matched, those words are consumed (no overlap).
        """
        if not text or not self._dictionary.reverse_index:
            return text

        # Split into tokens preserving whitespace structure
        parts = _TOKEN_RE.split(text)
        # parts alternates: [pre-space, word, space, word, space, word, ...]
        # Extract word tokens with their indices in `parts`
        word_indices: list[int] = []
        words: list[str] = []
        for i, part in enumerate(parts):
            if _TOKEN_RE.fullmatch(part):
                word_indices.append(i)
                words.append(part)

        if not words:
            return text

        # Strip leading/trailing punctuation from each word for matching purposes
        stripped_words: list[str] = []
        leading_punct: list[str] = []
        trailing_punct: list[str] = []
        for w in words:
            m = _PUNCT_STRIP_RE.match(w)
            if m:
                leading_punct.append(m.group(1))
                stripped_words.append(m.group(2))
                trailing_punct.append(m.group(3))
            else:
                leading_punct.append("")
                stripped_words.append(w)
                trailing_punct.append("")

        max_phrase = self._config.max_phrase_words
        consumed = [False] * len(words)
        replacements: list[tuple[int, int, str]] = []  # (start_word_idx, end_word_idx_exclusive, replacement)

        i = 0
        while i < len(words):
            if consumed[i]:
                i += 1
                continue

            matched = False
            # Try longest phrase first, shrink down to 1
            for phrase_len in range(min(max_phrase, len(words) - i), 0, -1):
                # Check none of the words in this window are already consumed
                if any(consumed[i + k] for k in range(phrase_len)):
                    continue

                phrase_words = stripped_words[i : i + phrase_len]
                phrase = " ".join(phrase_words)

                # Try exact match first
                canonical = self._dictionary.exact_lookup(phrase)
                if canonical is None:
                    # Try fuzzy match
                    canonical = self._dictionary.fuzzy_lookup(phrase, score_cutoff=self._config.fuzzy_threshold)

                # Phonetic fallback for single words >= 5 chars
                if canonical is None and phrase_len == 1 and len(phrase) >= 5:
                    canonical = self._dictionary.phonetic_lookup(phrase)

                if canonical is not None:
                    for k in range(phrase_len):
                        consumed[i + k] = True
                    # Reattach leading punct from first word, trailing punct from last word
                    replacement = leading_punct[i] + canonical + trailing_punct[i + phrase_len - 1]
                    replacements.append((i, i + phrase_len, replacement))
                    i += phrase_len
                    matched = True
                    break

            if not matched:
                i += 1

        # Apply replacements by rebuilding `parts`
        for start_word, end_word, replacement in replacements:
            # Replace the span from parts[word_indices[start_word]] to parts[word_indices[end_word-1]]
            # with the canonical form, collapsing intermediate whitespace
            first_part_idx = word_indices[start_word]
            last_part_idx = word_indices[end_word - 1]
            # Set the first token to the replacement, blank out the rest
            parts[first_part_idx] = replacement
            for pi in range(first_part_idx + 1, last_part_idx + 1):
                parts[pi] = ""

        return "".join(parts)
