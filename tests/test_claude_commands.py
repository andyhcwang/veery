"""Tests for Claude Code jargon mining: command scanning, variant generation, and correction."""

from __future__ import annotations

from pathlib import Path

import yaml

from veery.config import PROJECT_ROOT, JargonConfig
from veery.jargon import JargonCorrector, JargonDictionary
from veery.miner import (
    _generate_command_variants,
    _scan_claude_commands,
    mine_claude_commands,
    write_claude_commands_yaml,
)

# ---------------------------------------------------------------------------
# _generate_command_variants() unit tests
# ---------------------------------------------------------------------------


class TestGenerateCommandVariants:
    """Unit tests for _generate_command_variants()."""

    def test_single_word_command(self) -> None:
        """Single-word stem produces 'slash <word>' variant."""
        variants = _generate_command_variants("commit")
        assert "slash commit" in variants

    def test_single_word_no_bare_variant(self) -> None:
        """Single-word stem does NOT include bare word (would cause false positives)."""
        variants = _generate_command_variants("commit")
        assert "commit" not in variants

    def test_multi_word_command(self) -> None:
        """Multi-word stem produces both slash-prefixed and bare variants."""
        variants = _generate_command_variants("review-pr")
        # Should have slash-prefixed variant
        assert any(v.startswith("slash ") for v in variants)
        # Should have bare variant (for max_phrase_words=3 compliance)
        assert any(not v.startswith("slash ") for v in variants)

    def test_multi_word_bare_variant(self) -> None:
        """Multi-word commands include bare variants to stay within sliding window."""
        variants = _generate_command_variants("add-dir")
        assert "add dir" in variants

    def test_three_part_command(self) -> None:
        """Three-part command generates both slash-prefixed and bare variants."""
        variants = _generate_command_variants("commit-push-pr")
        assert any(not v.startswith("slash ") for v in variants)
        assert any(v.startswith("slash ") for v in variants)

    def test_phonetic_expansion(self) -> None:
        """Parts in _PHONETIC_SUBS get expanded (e.g., 'pr' -> 'p r')."""
        variants = _generate_command_variants("review-pr")
        # "pr" should expand to "p r" via _PHONETIC_SUBS
        assert any("p r" in v for v in variants)

    def test_env_phonetic_expansion(self) -> None:
        """'env' has a phonetic sub entry -> expanded."""
        variants = _generate_command_variants("remote-env")
        assert any("e n v" in v for v in variants)

    def test_mcp_phonetic_expansion(self) -> None:
        """'mcp' has a phonetic sub entry -> expanded."""
        variants = _generate_command_variants("mcp")
        assert any("m c p" in v for v in variants)

    def test_empty_stem(self) -> None:
        """Empty stem returns empty list."""
        assert _generate_command_variants("") == []

    def test_hyphen_only(self) -> None:
        """Stem of just hyphens returns empty list."""
        assert _generate_command_variants("-") == []
        assert _generate_command_variants("--") == []

    def test_sorted_output(self) -> None:
        """Output is sorted."""
        variants = _generate_command_variants("review-pr")
        assert variants == sorted(variants)

    def test_no_duplicates(self) -> None:
        """No duplicate variants."""
        variants = _generate_command_variants("review-pr")
        assert len(variants) == len(set(variants))

    def test_single_char_parts_handled(self) -> None:
        """Single-character parts in stem are handled without error."""
        variants = _generate_command_variants("a-b")
        assert len(variants) > 0


# ---------------------------------------------------------------------------
# _scan_claude_commands() unit tests
# ---------------------------------------------------------------------------


class TestScanClaudeCommands:
    """Tests for scanning .claude/commands/ directories."""

    def test_basic_command_scan(self, tmp_path: Path) -> None:
        """Finds .md files in .claude/commands/ and returns {canonical: [variants]}."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "review-pr.md").write_text("Review the PR")
        (cmd_dir / "deploy.md").write_text("Deploy to prod")

        commands = _scan_claude_commands(tmp_path)

        assert "/review-pr" in commands
        assert "/deploy" in commands
        assert len(commands) == 2

    def test_no_commands_dir(self, tmp_path: Path) -> None:
        """Returns empty dict when .claude/commands/ doesn't exist."""
        commands = _scan_claude_commands(tmp_path)
        assert commands == {}

    def test_empty_commands_dir(self, tmp_path: Path) -> None:
        """Returns empty dict when .claude/commands/ exists but is empty."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)

        commands = _scan_claude_commands(tmp_path)
        assert commands == {}

    def test_subdirectory_commands(self, tmp_path: Path) -> None:
        """Subdirectory commands use colon notation: /subdir:cmd."""
        cmd_dir = tmp_path / ".claude" / "commands"
        sub_dir = cmd_dir / "frontend"
        sub_dir.mkdir(parents=True)
        (sub_dir / "lint.md").write_text("Run frontend lint")

        commands = _scan_claude_commands(tmp_path)

        assert "/frontend:lint" in commands
        assert len(commands) == 1

    def test_nested_subdirectory(self, tmp_path: Path) -> None:
        """Deeply nested subdirectory commands use colon notation with all parts."""
        cmd_dir = tmp_path / ".claude" / "commands"
        nested = cmd_dir / "team" / "backend"
        nested.mkdir(parents=True)
        (nested / "test.md").write_text("Run backend tests")

        commands = _scan_claude_commands(tmp_path)

        assert "/team:backend:test" in commands

    def test_skips_non_md_files(self, tmp_path: Path) -> None:
        """Non-.md files are ignored."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "deploy.md").write_text("Deploy")
        (cmd_dir / "notes.txt").write_text("Notes")
        (cmd_dir / "script.py").write_text("print('hi')")

        commands = _scan_claude_commands(tmp_path)

        assert "/deploy" in commands
        assert len(commands) == 1

    def test_skips_readme(self, tmp_path: Path) -> None:
        """README.md files are skipped."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "README.md").write_text("# Commands")
        (cmd_dir / "deploy.md").write_text("Deploy")

        commands = _scan_claude_commands(tmp_path)

        assert "/deploy" in commands
        assert len(commands) == 1  # README not counted

    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        """Hidden files (starting with '.') are skipped."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / ".hidden.md").write_text("hidden")
        (cmd_dir / "visible.md").write_text("visible")

        commands = _scan_claude_commands(tmp_path)

        assert "/visible" in commands
        assert len(commands) == 1

    def test_skips_short_stems(self, tmp_path: Path) -> None:
        """Single-character stems are skipped (too short)."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "a.md").write_text("too short")
        (cmd_dir / "ok.md").write_text("fine")

        commands = _scan_claude_commands(tmp_path)

        assert "/ok" in commands
        assert len(commands) == 1

    def test_commands_have_variants(self, tmp_path: Path) -> None:
        """Each scanned command has non-empty variant list."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "deploy.md").write_text("Deploy")

        commands = _scan_claude_commands(tmp_path)

        assert "/deploy" in commands
        assert len(commands["/deploy"]) > 0
        assert all(isinstance(v, str) for v in commands["/deploy"])

    def test_multiple_commands_mixed(self, tmp_path: Path) -> None:
        """Mix of top-level and subdirectory commands."""
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "commit.md").write_text("Commit")

        sub_dir = cmd_dir / "ci"
        sub_dir.mkdir()
        (sub_dir / "deploy.md").write_text("Deploy")

        commands = _scan_claude_commands(tmp_path)

        assert "/commit" in commands
        assert "/ci:deploy" in commands
        assert len(commands) == 2


# ---------------------------------------------------------------------------
# mine_claude_commands() tests
# ---------------------------------------------------------------------------


class TestMineClaudeCommands:
    def test_scans_project_paths(self, tmp_path: Path) -> None:
        """mine_claude_commands scans each path's .claude/commands/."""
        project = tmp_path / "myproject"
        cmd_dir = project / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "review.md").write_text("Review code")

        config = JargonConfig(dict_paths=(), learned_path=None)
        commands = mine_claude_commands([project], config=config)

        assert "/review" in commands

    def test_filters_already_known(self, tmp_path: Path) -> None:
        """Commands already in existing jargon dicts are filtered out."""
        project = tmp_path / "myproject"
        cmd_dir = project / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "review.md").write_text("Review code")

        # Create a jargon dict that already has /review
        dict_path = tmp_path / "existing.yaml"
        dict_path.write_text(yaml.dump({"terms": {"/review": ["slash review"]}}))

        config = JargonConfig(dict_paths=(str(dict_path),), learned_path=None)
        commands = mine_claude_commands([project], config=config)

        assert "/review" not in commands

    def test_empty_scan_paths(self) -> None:
        """Empty scan_paths list returns empty dict (ignoring home dir)."""
        config = JargonConfig(dict_paths=(), learned_path=None)
        # This will also try to scan ~/.claude/commands, but that's OK for testing
        commands = mine_claude_commands([], config=config)
        # Should be a dict (possibly empty or with home commands)
        assert isinstance(commands, dict)

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent scan path is handled gracefully."""
        config = JargonConfig(dict_paths=(), learned_path=None)
        commands = mine_claude_commands([tmp_path / "nonexistent"], config=config)
        assert isinstance(commands, dict)


# ---------------------------------------------------------------------------
# write_claude_commands_yaml() tests
# ---------------------------------------------------------------------------


class TestWriteClaudeCommandsYaml:
    def test_writes_commands(self, tmp_path: Path) -> None:
        """Writes commands to a YAML file."""
        output = tmp_path / "commands.yaml"
        commands = {"/review-pr": ["slash review pr", "review p r"]}

        count = write_claude_commands_yaml(commands, output)

        assert count == 1
        assert output.exists()
        data = yaml.safe_load(output.read_text())
        assert "/review-pr" in data["terms"]
        assert data["terms"]["/review-pr"] == ["slash review pr", "review p r"]

    def test_empty_commands(self, tmp_path: Path) -> None:
        """Empty commands dict returns 0 and doesn't create file."""
        output = tmp_path / "commands.yaml"

        count = write_claude_commands_yaml({}, output)

        assert count == 0
        assert not output.exists()

    def test_merges_with_existing(self, tmp_path: Path) -> None:
        """Re-running merges without overwriting existing entries."""
        output = tmp_path / "commands.yaml"

        # First write
        commands1 = {"/deploy": ["slash deploy"]}
        write_claude_commands_yaml(commands1, output)

        # Manually edit the existing entry
        data = yaml.safe_load(output.read_text())
        data["terms"]["/deploy"] = ["my custom variant"]
        with open(output, "w") as f:
            yaml.dump(data, f)

        # Second write with both old and new commands
        commands2 = {"/deploy": ["slash deploy"], "/review": ["slash review"]}
        count = write_claude_commands_yaml(commands2, output)

        assert count == 1  # only /review is new
        data = yaml.safe_load(output.read_text())
        assert data["terms"]["/deploy"] == ["my custom variant"]  # preserved
        assert "/review" in data["terms"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        output = tmp_path / "deep" / "nested" / "commands.yaml"
        commands = {"/test": ["slash test"]}

        count = write_claude_commands_yaml(commands, output)

        assert count == 1
        assert output.exists()

    def test_header_comment(self, tmp_path: Path) -> None:
        """Output file includes a header comment."""
        output = tmp_path / "commands.yaml"
        commands = {"/deploy": ["slash deploy"]}

        write_claude_commands_yaml(commands, output)

        content = output.read_text()
        assert content.startswith("# Auto-generated by Veery")
        assert "Claude Code" in content

    def test_valid_yaml(self, tmp_path: Path) -> None:
        """Generated file is valid YAML loadable by JargonDictionary."""
        output = tmp_path / "commands.yaml"
        commands = {
            "/review-pr": ["slash review pr", "review p r"],
            "/deploy": ["slash deploy"],
        }
        write_claude_commands_yaml(commands, output)

        config = JargonConfig(dict_paths=(str(output),), learned_path=None)
        d = JargonDictionary(config)
        assert "slash review pr" in d.reverse_index
        assert d.reverse_index["slash review pr"] == "/review-pr"
        assert "slash deploy" in d.reverse_index

    def test_returns_zero_when_all_exist(self, tmp_path: Path) -> None:
        """If all commands already exist in file, returns 0."""
        output = tmp_path / "commands.yaml"
        commands = {"/deploy": ["slash deploy"]}

        write_claude_commands_yaml(commands, output)
        count = write_claude_commands_yaml(commands, output)

        assert count == 0


# ---------------------------------------------------------------------------
# claude_code.yaml built-in loading tests
# ---------------------------------------------------------------------------


class TestClaudeCodeYamlLoading:
    """Tests for the built-in jargon/claude_code.yaml file."""

    def test_claude_code_yaml_exists(self) -> None:
        """The built-in claude_code.yaml file exists."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        assert path.exists()

    def test_claude_code_yaml_valid(self) -> None:
        """The built-in claude_code.yaml is valid YAML with expected structure."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        data = yaml.safe_load(path.read_text())
        assert "terms" in data
        assert isinstance(data["terms"], dict)
        assert len(data["terms"]) > 10  # should have ~40 commands

    def test_claude_code_yaml_loadable(self) -> None:
        """The built-in claude_code.yaml loads correctly into JargonDictionary."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        d = JargonDictionary(config)
        assert len(d.reverse_index) > 0

    def test_slash_commands_in_index(self) -> None:
        """Slash command variants map to correct canonical forms."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        d = JargonDictionary(config)

        assert d.reverse_index.get("slash commit") == "/commit"
        assert d.reverse_index.get("slash help") == "/help"
        assert d.reverse_index.get("slash review") == "/review"

    def test_at_mentions_in_index(self) -> None:
        """@-mention variants map to correct canonical forms."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        d = JargonDictionary(config)

        assert d.reverse_index.get("at docs") == "@docs"
        assert d.reverse_index.get("at codebase") == "@codebase"

    def test_multi_word_commands_in_index(self) -> None:
        """Multi-word slash commands have proper variants."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        d = JargonDictionary(config)

        # /add-dir has variants like "slash add dir" and "add dir"
        assert d.reverse_index.get("slash add dir") == "/add-dir"
        assert d.reverse_index.get("add dir") == "/add-dir"


# ---------------------------------------------------------------------------
# End-to-end jargon correction tests
# ---------------------------------------------------------------------------


class TestClaudeCommandCorrection:
    """End-to-end: dictated text -> jargon corrector -> correct slash commands."""

    @staticmethod
    def _make_corrector() -> JargonCorrector:
        """Create a JargonCorrector with only claude_code.yaml loaded."""
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        return JargonCorrector(config)

    def test_slash_commit(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("slash commit")
        assert result == "/commit"

    def test_slash_help(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("slash help")
        assert result == "/help"

    def test_slash_review(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("slash review")
        assert result == "/review"

    def test_at_docs(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("at docs")
        assert result == "@docs"

    def test_at_codebase(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("at codebase")
        assert result == "@codebase"

    def test_multi_word_add_dir(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("slash add dir")
        assert result == "/add-dir"

    def test_bare_multi_word_add_dir(self) -> None:
        """Bare multi-word variant 'add dir' corrects to /add-dir."""
        corrector = self._make_corrector()
        result = corrector.correct("add dir")
        assert result == "/add-dir"

    def test_slash_in_sentence(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("run slash commit now")
        assert "/commit" in result
        assert "run" in result
        assert "now" in result

    def test_plain_slash_no_match(self) -> None:
        """The word 'slash' alone should not trigger a match."""
        corrector = self._make_corrector()
        result = corrector.correct("slash")
        assert result == "slash"

    def test_unrelated_text_unchanged(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("the quick brown fox")
        assert result == "the quick brown fox"

    def test_pr_comments_correction(self) -> None:
        """Multi-word command /pr-comments has variant with phonetic expansion."""
        corrector = self._make_corrector()
        result = corrector.correct("p r comments")
        assert result == "/pr-comments"

    def test_output_style_correction(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("output style")
        assert result == "/output-style"


# ---------------------------------------------------------------------------
# False positive resistance tests
# ---------------------------------------------------------------------------


class TestClaudeCommandFalsePositives:
    """Commands should not trigger on unrelated text."""

    @staticmethod
    def _make_corrector() -> JargonCorrector:
        path = PROJECT_ROOT / "jargon" / "claude_code.yaml"
        config = JargonConfig(dict_paths=(str(path),), learned_path=None)
        return JargonCorrector(config)

    def test_common_words_unchanged(self) -> None:
        corrector = self._make_corrector()
        for text in [
            "I need help with this",
            "please review the document",
            "clear the table",
            "commit to the plan",
            "exit the building",
        ]:
            # These contain words that are command stems but without "slash" prefix
            # Single-word bare variants are NOT generated, so these should pass through
            result = corrector.correct(text)
            assert "/" not in result, f"False positive for: {text!r} -> {result!r}"

    def test_plain_slash_alone(self) -> None:
        corrector = self._make_corrector()
        assert corrector.correct("slash") == "slash"

    def test_slash_with_unknown_word(self) -> None:
        corrector = self._make_corrector()
        result = corrector.correct("slash banana")
        # "slash banana" should not match any command
        assert result == "slash banana"


# ---------------------------------------------------------------------------
# Config default dict_paths tests
# ---------------------------------------------------------------------------


class TestConfigDictPaths:
    """Verify claude_code.yaml is in default dict_paths."""

    def test_claude_code_in_default_paths(self) -> None:
        config = JargonConfig()
        assert "jargon/claude_code.yaml" in config.dict_paths

    def test_mined_commands_in_default_paths(self) -> None:
        config = JargonConfig()
        assert "jargon/mined_commands.yaml" in config.dict_paths

    def test_default_paths_count(self) -> None:
        config = JargonConfig()
        assert len(config.dict_paths) == 5
