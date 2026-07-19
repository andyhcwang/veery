"""Accuracy evaluation: mixed Chinese/English error rate over dictation history.

The metric is a mixed error rate (MER): CJK text is tokenized per character,
Latin text per lowercased word, punctuation is ignored. MER is the token-level
edit distance divided by the reference length — the standard CER/WER hybrid
for code-switched speech.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_LATIN_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-z]+)?")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize code-switched text: CJK per character, Latin per word.

    Punctuation and whitespace are dropped; Latin is lowercased so
    capitalization differences don't count as errors.
    """
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _CJK_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        m = _LATIN_WORD_RE.match(text, i)
        if m:
            tokens.append(m.group(0).lower())
            i = m.end()
            continue
        i += 1  # punctuation/space/other: skip
    return tokens


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token lists (O(len(a)*len(b)))."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def mixed_error_rate(hypothesis: str, reference: str) -> float:
    """Token-level edit distance / reference length (0.0 = perfect).

    Can exceed 1.0 when the hypothesis is much longer than the reference.
    An empty reference scores 0.0 for an empty hypothesis, else 1.0.
    """
    ref = tokenize_mixed(reference)
    hyp = tokenize_mixed(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(hyp, ref) / len(ref)


@dataclass
class EvalResult:
    name: str
    n_samples: int
    error_rate: float  # sample-weighted mean MER

    def __str__(self) -> str:
        return f"{self.name:<24} n={self.n_samples:<4} MER={self.error_rate * 100:6.2f}%"


def evaluate_pairs(pairs: list[tuple[str, str]], name: str = "all") -> EvalResult:
    """Score (hypothesis, reference) pairs; weighted by reference token count."""
    total_tokens = 0
    total_errors = 0.0
    for hyp, ref in pairs:
        ref_tokens = tokenize_mixed(ref)
        if not ref_tokens:
            continue
        total_tokens += len(ref_tokens)
        total_errors += mixed_error_rate(hyp, ref) * len(ref_tokens)
    rate = (total_errors / total_tokens) if total_tokens else 0.0
    return EvalResult(name=name, n_samples=len(pairs), error_rate=rate)


def match_script_line(hypothesis: str, script_lines: list[str]) -> tuple[str, float] | None:
    """Find the benchmark script line a recording most likely reads.

    Returns (best_line, error_rate_against_it), or None when nothing is
    plausibly close (MER > 0.8 — likely free dictation, not the script).
    """
    best: tuple[str, float] | None = None
    for line in script_lines:
        rate = mixed_error_rate(hypothesis, line)
        if best is None or rate < best[1]:
            best = (line, rate)
    if best is None or best[1] > 0.8:
        return None
    return best


def _parse_script(path: str) -> list[tuple[str, str]]:
    """Parse a benchmark script into (category, line) pairs.

    Lines starting with '#' set the current category; blank lines skipped.
    """
    pairs: list[tuple[str, str]] = []
    category = "uncategorized"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                category = line.lstrip("#").strip() or category
                continue
            pairs.append((category, line))
    return pairs


def run_eval(config, script_path: str | None = None) -> None:
    """CLI entry: score dictation history accuracy.

    Two label sources:
    - user corrections attached to history records (manual edits / re-dictation)
    - a benchmark reading script: each record is matched to its closest line
    """
    from veery.history import DictationHistory

    history = DictationHistory(config.history)
    records = history.load_records()
    if not records:
        print("No dictation history found.")
        print("Enable it in config.yaml (history: enabled: true), dictate a while, then re-run.")
        return

    corrected = [(r.final_text, r.corrected_text) for r in records if r.corrected_text]
    print(f"History: {len(records)} records, {len(corrected)} with user-correction labels\n")

    if corrected:
        print(evaluate_pairs(corrected, "user-corrections"))

    if script_path:
        pairs = _parse_script(script_path)
        if not pairs:
            print(f"No lines found in {script_path}")
            return
        lines = [line for _, line in pairs]
        category_of = {line: cat for cat, line in pairs}
        by_category: dict[str, list[tuple[str, str]]] = {}
        matched = 0
        for r in records:
            best = match_script_line(r.final_text, lines)
            if best is None:
                continue
            matched += 1
            line, _ = best
            by_category.setdefault(category_of[line], []).append((r.final_text, line))
        print(f"\nBenchmark script: {matched}/{len(records)} records matched a script line")
        all_pairs: list[tuple[str, str]] = []
        for cat, cat_pairs in by_category.items():
            print(evaluate_pairs(cat_pairs, cat))
            all_pairs.extend(cat_pairs)
        if all_pairs:
            print(evaluate_pairs(all_pairs, "OVERALL"))

    if not corrected and not script_path:
        print("No labels yet. Labels come from:")
        print("  1. Editing pasted text right after dictating (auto-captured)")
        print("  2. Reading benchmark/script.txt aloud, then: veery --eval-script benchmark/script.txt")
