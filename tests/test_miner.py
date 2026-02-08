"""Tests for jargon mining from Python source code."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from voiceflow.config import JargonConfig
from voiceflow.miner import _is_interesting_name, _extract_names_from_ast, _scan_directory, mine_terms


class TestExtractAllCaps:
    def test_extract_all_caps(self, tmp_path: Path) -> None:
        """Python file with TWAP = '...' extracts 'TWAP'."""
        py_file = tmp_path / "constants.py"
        py_file.write_text('TWAP = "time-weighted average price"\nVWAP = "volume-weighted"\n')

        names = _extract_names_from_ast(py_file.read_text())
        assert "TWAP" in names
        assert "VWAP" in names


class TestExtractCamelCase:
    def test_extract_camelcase(self, tmp_path: Path) -> None:
        """Python file with class MyClassName extracts 'MyClassName'."""
        py_file = tmp_path / "classes.py"
        py_file.write_text("class MyClassName:\n    pass\n\nclass DuckDb:\n    pass\n")

        names = _extract_names_from_ast(py_file.read_text())
        assert "MyClassName" in names
        assert "DuckDb" in names


class TestSkipTestPrefix:
    def test_skip_test_prefix(self, tmp_path: Path) -> None:
        """class TestSomething and testHelper not extracted (test prefix filtering)."""
        py_file = tmp_path / "test_stuff.py"
        py_file.write_text(
            "class TestSomething:\n    pass\n\n"
            "class testHelper:\n    pass\n\n"
        )

        names = _extract_names_from_ast(py_file.read_text())
        assert "TestSomething" not in names
        assert "testHelper" not in names

    def test_all_caps_test_prefix_kept(self, tmp_path: Path) -> None:
        """ALL_CAPS names starting with TEST_ are kept (they're valid constants)."""
        py_file = tmp_path / "consts.py"
        py_file.write_text("TEST_VAR = 1\n")

        names = _extract_names_from_ast(py_file.read_text())
        # ALL_CAPS regex matches before test prefix check, so TEST_VAR is kept
        assert "TEST_VAR" in names


class TestSkipDunder:
    def test_skip_dunder(self, tmp_path: Path) -> None:
        """__init__, __name__ not extracted."""
        py_file = tmp_path / "module.py"
        py_file.write_text("__init__ = None\n__name__ = 'test'\n__all__ = ['foo']\n")

        names = _extract_names_from_ast(py_file.read_text())
        assert "__init__" not in names
        assert "__name__" not in names
        assert "__all__" not in names


class TestSkipShortNames:
    def test_skip_short_names(self, tmp_path: Path) -> None:
        """1-2 char names not extracted."""
        py_file = tmp_path / "short.py"
        py_file.write_text("A = 1\nXY = 2\nclass Ab:\n    pass\n")

        names = _extract_names_from_ast(py_file.read_text())
        # Single-char and 2-char names should be filtered
        assert "A" not in names
        assert "XY" not in names
        assert "Ab" not in names


class TestFrequencyCounting:
    def test_frequency_counting(self, tmp_path: Path) -> None:
        """Same name in multiple files counted correctly."""
        dir1 = tmp_path / "pkg"
        dir1.mkdir()
        (dir1 / "a.py").write_text("TWAP = 1\n")
        (dir1 / "b.py").write_text("TWAP = 2\nVWAP = 3\n")
        (dir1 / "c.py").write_text("TWAP = 3\n")

        counter = _scan_directory(dir1)
        assert counter["TWAP"] == 3
        assert counter["VWAP"] == 1


class TestSkipExcludedDirs:
    def test_skip_excluded_dirs(self, tmp_path: Path) -> None:
        """Files in .venv/ or __pycache__/ skipped."""
        # Create a file in a skippable directory
        venv_dir = tmp_path / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "module.py").write_text("SHOULD_NOT_SEE = 1\n")

        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("ALSO_HIDDEN = 1\n")

        # Create a normal file
        (tmp_path / "real.py").write_text("VISIBLE = 1\n")

        counter = _scan_directory(tmp_path)
        assert "SHOULD_NOT_SEE" not in counter
        assert "ALSO_HIDDEN" not in counter
        assert "VISIBLE" in counter


class TestAlreadyKnownMarking:
    def test_already_known_marking(self, tmp_path: Path) -> None:
        """Terms in existing YAML marked as 'already known'."""
        # Create a jargon YAML with a known term
        dict_path = tmp_path / "jargon.yaml"
        dict_path.write_text(yaml.dump({"terms": {"TWAP": ["tee wap"]}}))

        # Create a Python file that uses TWAP
        code_dir = tmp_path / "src"
        code_dir.mkdir()
        (code_dir / "algo.py").write_text("TWAP = 'time weighted'\nVWAP = 'vol weighted'\n")

        config = JargonConfig(dict_paths=(str(dict_path),), learned_path=None)
        results = mine_terms([code_dir], config=config)

        result_dict = {term: (freq, known) for term, freq, known in results}
        assert result_dict["TWAP"] == (1, True)
        assert result_dict["VWAP"] == (1, False)


class TestMineEmptyDirectory:
    def test_mine_empty_directory(self, tmp_path: Path) -> None:
        """Empty dir returns empty results."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = JargonConfig(dict_paths=(), learned_path=None)
        results = mine_terms([empty_dir], config=config)
        assert results == []


class TestIsInterestingName:
    """Unit tests for the _is_interesting_name filter function."""

    def test_all_caps_accepted(self) -> None:
        assert _is_interesting_name("TWAP") is True
        assert _is_interesting_name("VWAP") is True
        assert _is_interesting_name("GMV") is True

    def test_camel_case_accepted(self) -> None:
        assert _is_interesting_name("MyClassName") is True
        assert _is_interesting_name("DuckDb") is True
        assert _is_interesting_name("PyTorch") is True

    def test_short_rejected(self) -> None:
        assert _is_interesting_name("A") is False
        assert _is_interesting_name("XY") is False

    def test_underscore_prefix_rejected(self) -> None:
        assert _is_interesting_name("_PRIVATE") is False
        assert _is_interesting_name("__init__") is False

    def test_test_prefix_rejected(self) -> None:
        assert _is_interesting_name("TestCase") is False
        assert _is_interesting_name("test_something") is False
        # Note: ALL_CAPS TEST_VAR passes because _ALL_CAPS_RE matches
        # before the test prefix check. This is by design.
        assert _is_interesting_name("TEST_VAR") is True

    def test_setup_teardown_rejected(self) -> None:
        assert _is_interesting_name("setUp") is False
        assert _is_interesting_name("tearDown") is False

    def test_lowercase_rejected(self) -> None:
        assert _is_interesting_name("foobar") is False
        assert _is_interesting_name("hello") is False

    def test_all_caps_with_digits(self) -> None:
        assert _is_interesting_name("K8S") is True
        assert _is_interesting_name("S3_BUCKET") is True
