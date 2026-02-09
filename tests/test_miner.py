"""Tests for jargon mining from Python source code."""

from __future__ import annotations

from pathlib import Path

import yaml

from veery.config import JargonConfig
from veery.miner import (
    _extract_names_from_ast,
    _is_interesting_name,
    _scan_directory,
    _split_camel_case,
    generate_variants,
    mine_terms,
    write_mined_yaml,
)


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


class TestSplitCamelCase:
    """Unit tests for _split_camel_case()."""

    def test_standard_camel(self) -> None:
        assert _split_camel_case("SvelteKit") == ["Svelte", "Kit"]

    def test_trailing_acronym(self) -> None:
        assert _split_camel_case("DuckDB") == ["Duck", "DB"]

    def test_leading_acronym(self) -> None:
        assert _split_camel_case("APIKey") == ["API", "Key"]

    def test_lowercase_prefix(self) -> None:
        assert _split_camel_case("vLLM") == ["v", "LLM"]

    def test_all_caps(self) -> None:
        assert _split_camel_case("TWAP") == ["TWAP"]

    def test_two_word(self) -> None:
        assert _split_camel_case("PyTorch") == ["Py", "Torch"]

    def test_three_word(self) -> None:
        assert _split_camel_case("MyClassName") == ["My", "Class", "Name"]

    def test_mixed_acronym_and_words(self) -> None:
        assert _split_camel_case("FastAPI") == ["Fast", "API"]

    def test_digits_in_name(self) -> None:
        assert _split_camel_case("Vue3Router") == ["Vue3", "Router"]

    def test_single_word(self) -> None:
        assert _split_camel_case("Router") == ["Router"]


class TestGenerateVariants:
    """Unit tests for generate_variants()."""

    def test_camel_case_split(self) -> None:
        variants = generate_variants("SvelteKit")
        assert "svelte kit" in variants

    def test_lowercase_flattening(self) -> None:
        variants = generate_variants("SvelteKit")
        assert "sveltekit" in variants

    def test_acronym_letter_spacing(self) -> None:
        variants = generate_variants("TWAP")
        assert "t w a p" in variants

    def test_no_letter_spacing_for_long_all_caps(self) -> None:
        # ALL_CAPS >5 chars should not get letter-spaced
        variants = generate_variants("BUFFER")
        assert "b u f f e r" not in variants

    def test_phonetic_substitution(self) -> None:
        variants = generate_variants("PyTorch")
        assert "pie torch" in variants

    def test_no_self_variant_for_single_part(self) -> None:
        # Single-part terms should not include themselves lowered as a variant
        for term in ["TWAP", "Router"]:
            variants = generate_variants(term)
            assert term.lower() not in variants

    def test_flat_form_kept_for_multi_part(self) -> None:
        # Multi-part CamelCase terms DO include the flat form as a variant
        assert "pytorch" in generate_variants("PyTorch")
        assert "sveltekit" in generate_variants("SvelteKit")

    def test_sorted_output(self) -> None:
        variants = generate_variants("DuckDB")
        assert variants == sorted(variants)

    def test_dedup(self) -> None:
        variants = generate_variants("SvelteKit")
        assert len(variants) == len(set(variants))

    def test_pytorch_variants(self) -> None:
        variants = generate_variants("PyTorch")
        assert "pie torch" in variants
        assert "py torch" in variants
        assert "pytorch" in variants

    def test_fastapi_variants(self) -> None:
        variants = generate_variants("FastAPI")
        assert "fast a p i" in variants
        assert "fast api" in variants
        assert "fastapi" in variants


class TestWriteMinedYaml:
    """Tests for write_mined_yaml()."""

    def test_writes_new_terms(self, tmp_path: Path) -> None:
        """New terms are written to a YAML file with variants."""
        output = tmp_path / "mined.yaml"
        results = [("PyTorch", 5, False), ("TWAP", 3, False)]
        count = write_mined_yaml(results, output, [tmp_path])

        assert count == 2
        assert output.exists()
        with open(output) as f:
            content = f.read()
        data = yaml.safe_load(content)
        assert "PyTorch" in data["terms"]
        assert "TWAP" in data["terms"]
        assert isinstance(data["terms"]["PyTorch"], list)

    def test_skips_known_terms(self, tmp_path: Path) -> None:
        """Already-known terms are not written."""
        output = tmp_path / "mined.yaml"
        results = [("PyTorch", 5, True), ("TWAP", 3, False)]
        count = write_mined_yaml(results, output, [tmp_path])

        assert count == 1
        data = yaml.safe_load(output.read_text())
        assert "PyTorch" not in data["terms"]
        assert "TWAP" in data["terms"]

    def test_returns_zero_for_all_known(self, tmp_path: Path) -> None:
        """If all terms are known, returns 0 and doesn't create file."""
        output = tmp_path / "mined.yaml"
        results = [("PyTorch", 5, True), ("TWAP", 3, True)]
        count = write_mined_yaml(results, output, [tmp_path])

        assert count == 0
        assert not output.exists()

    def test_merges_with_existing(self, tmp_path: Path) -> None:
        """Re-running merges without overwriting existing terms."""
        output = tmp_path / "mined.yaml"
        # First run: write PyTorch
        results1 = [("PyTorch", 5, False)]
        write_mined_yaml(results1, output, [tmp_path])

        # Edit PyTorch variants by hand
        data = yaml.safe_load(output.read_text())
        data["terms"]["PyTorch"] = ["my custom variant"]
        with open(output, "w") as f:
            yaml.dump(data, f)

        # Second run: add TWAP, should NOT overwrite PyTorch
        results2 = [("PyTorch", 5, False), ("TWAP", 3, False)]
        count = write_mined_yaml(results2, output, [tmp_path])

        assert count == 1  # only TWAP is new
        data = yaml.safe_load(output.read_text())
        assert data["terms"]["PyTorch"] == ["my custom variant"]
        assert "TWAP" in data["terms"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        output = tmp_path / "deep" / "nested" / "mined.yaml"
        results = [("TWAP", 3, False)]
        count = write_mined_yaml(results, output, [tmp_path])

        assert count == 1
        assert output.exists()

    def test_header_comment(self, tmp_path: Path) -> None:
        """Output file includes a header comment."""
        output = tmp_path / "mined.yaml"
        results = [("TWAP", 3, False)]
        write_mined_yaml(results, output, [tmp_path])

        content = output.read_text()
        assert content.startswith("# Auto-generated by Veery")
        assert "Edit freely" in content

    def test_loadable_yaml(self, tmp_path: Path) -> None:
        """Generated file is valid YAML loadable by JargonDictionary."""
        output = tmp_path / "mined.yaml"
        results = [("PyTorch", 5, False), ("DuckDB", 3, False), ("TWAP", 2, False)]
        write_mined_yaml(results, output, [tmp_path])

        # Verify it can be loaded as a jargon dictionary
        from veery.jargon import JargonCorrector
        config = JargonConfig(dict_paths=(str(output),), learned_path=None)
        corrector = JargonCorrector(config)
        assert corrector is not None

    def test_empty_results(self, tmp_path: Path) -> None:
        """Empty results returns 0 and doesn't create file."""
        output = tmp_path / "mined.yaml"
        count = write_mined_yaml([], output, [tmp_path])

        assert count == 0
        assert not output.exists()
