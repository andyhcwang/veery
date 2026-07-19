"""Tests for accuracy evaluation and dictation history."""

from __future__ import annotations

import numpy as np

from veery.config import HistoryConfig
from veery.evaluate import (
    evaluate_pairs,
    match_script_line,
    mixed_error_rate,
    tokenize_mixed,
)
from veery.history import DictationHistory


class TestTokenizeMixed:
    def test_cjk_per_char_latin_per_word(self) -> None:
        assert tokenize_mixed("帮我review这个PR") == ["帮", "我", "review", "这", "个", "pr"]

    def test_punctuation_dropped_case_folded(self) -> None:
        assert tokenize_mixed("Hello, World! 你好。") == ["hello", "world", "你", "好"]

    def test_numbers_kept(self) -> None:
        assert tokenize_mixed("等3秒") == ["等", "3", "秒"]


class TestMixedErrorRate:
    def test_perfect_match_zero(self) -> None:
        assert mixed_error_rate("帮我review这个PR", "帮我 review 这个 PR!") == 0.0

    def test_one_char_substitution(self) -> None:
        # 胡六句话 vs 五六句话: 1 error over 4 CJK tokens
        assert mixed_error_rate("胡六句话", "五六句话") == 0.25

    def test_empty_hypothesis_full_error(self) -> None:
        assert mixed_error_rate("", "你好") == 1.0

    def test_both_empty_perfect(self) -> None:
        assert mixed_error_rate("", "") == 0.0


class TestScriptMatching:
    LINES = ["今天的心情比昨天晴朗很多", "The latency spike was caused by a bug."]

    def test_close_match_found(self) -> None:
        best = match_script_line("今天的心情比昨天晴朗很多了", self.LINES)
        assert best is not None
        assert best[0] == self.LINES[0]

    def test_unrelated_text_returns_none(self) -> None:
        assert match_script_line("completely different words here entirely", self.LINES) is None

    def test_evaluate_pairs_weighted(self) -> None:
        result = evaluate_pairs([("五六句话", "五六句话"), ("胡六句话", "五六句话")], "t")
        assert result.n_samples == 2
        assert abs(result.error_rate - 0.125) < 1e-9  # 1 error / 8 ref tokens


class TestDictationHistory:
    def _history(self, tmp_path, max_records: int = 500) -> DictationHistory:
        return DictationHistory(
            HistoryConfig(enabled=True, dir=str(tmp_path / "hist"), max_records=max_records)
        )

    def _audio(self) -> np.ndarray:
        return np.zeros(1600, dtype=np.float32)

    def test_save_and_load(self, tmp_path) -> None:
        h = self._history(tmp_path)
        h.save(self._audio(), 16000, "raw text", "final text")
        records = h.load_records()
        assert len(records) == 1
        assert records[0].final_text == "final text"
        assert records[0].wav_path.is_file()
        assert records[0].corrected_text is None

    def test_log_correction_labels_matching_record(self, tmp_path) -> None:
        h = self._history(tmp_path)
        h.save(self._audio(), 16000, "raw", "胡六句话")
        h.log_correction("胡六句话", "五六句话")
        records = h.load_records()
        assert records[0].corrected_text == "五六句话"

    def test_log_correction_ignores_unrelated_text(self, tmp_path) -> None:
        h = self._history(tmp_path)
        h.save(self._audio(), 16000, "raw", "some text")
        h.log_correction("different text", "corrected")
        assert h.load_records()[0].corrected_text is None

    def test_cap_deletes_oldest(self, tmp_path) -> None:
        h = self._history(tmp_path, max_records=10)
        for i in range(13):
            h.save(self._audio(), 16000, f"raw{i}", f"final{i}")
        records = h.load_records()
        assert len(records) == 10
        assert records[0].final_text == "final3"  # oldest three dropped
        # dropped wavs are gone from disk
        wavs = list((tmp_path / "hist").glob("*.wav"))
        assert len(wavs) == 10

    def test_save_never_raises_on_bad_dir(self) -> None:
        h = DictationHistory(HistoryConfig(enabled=True, dir="/dev/null/impossible"))
        h.save(self._audio(), 16000, "a", "b")  # must not raise
        assert h.load_records() == []
