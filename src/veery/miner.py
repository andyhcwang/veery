"""Bootstrap jargon mining from Python source code."""

from __future__ import annotations

import ast
import itertools
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from veery.config import PROJECT_ROOT, JargonConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# CamelCase splitting
# ──────────────────────────────────────────────────────────────────

_CAMEL_SPLIT_RE1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_SPLIT_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _split_camel_case(name: str) -> list[str]:
    """Split a CamelCase/mixed-case name into word parts.

    Examples:
        SvelteKit  -> ["Svelte", "Kit"]
        DuckDB     -> ["Duck", "DB"]
        PyTorch    -> ["Py", "Torch"]
        APIKey     -> ["API", "Key"]
        vLLM       -> ["v", "LLM"]
        TWAP       -> ["TWAP"]
    """
    s = _CAMEL_SPLIT_RE1.sub(r"\1 \2", name)
    s = _CAMEL_SPLIT_RE2.sub(r"\1 \2", s)
    return s.split()


# ──────────────────────────────────────────────────────────────────
# Phonetic substitution table
# ──────────────────────────────────────────────────────────────────

_PHONETIC_SUBS: dict[str, list[str]] = {
    "py": ["pie"],
    "db": ["dee bee"],
    "js": ["j s"],
    "ts": ["t s"],
    "ml": ["m l"],
    "ai": ["a i"],
    "sql": ["s q l", "sequel"],
    "api": ["a p i"],
    "llm": ["l l m"],
    "cli": ["c l i"],
    "gpu": ["g p u"],
    "cpu": ["c p u"],
    "npm": ["n p m"],
    "aws": ["a w s"],
    "gcp": ["g c p"],
    "ssh": ["s s h"],
    "url": ["u r l"],
    "gui": ["g u i", "gooey"],
    "ux": ["u x"],
    "ui": ["u i"],
    "ci": ["c i"],
    "cd": ["c d"],
    "qa": ["q a"],
    "os": ["o s"],
    "io": ["i o"],
    "vm": ["v m"],
    "k8s": ["k 8 s", "kubernetes"],
    "pr": ["p r"],
    "env": ["e n v"],
    "mcp": ["m c p"],
}


def _expand_phonetic_variants(parts: list[str]) -> list[str]:
    """Generate phonetic variants by substituting known fragments.

    Applies _PHONETIC_SUBS to each part (case-insensitive), then builds the
    cartesian product.  Capped at 8 results to prevent explosion.
    """
    options_per_part: list[list[str]] = []
    for part in parts:
        key = part.lower()
        subs = _PHONETIC_SUBS.get(key)
        options_per_part.append([key] + subs if subs else [key])

    return [
        " ".join(combo)
        for combo in itertools.islice(itertools.product(*options_per_part), 8)
    ]


# ──────────────────────────────────────────────────────────────────
# Public variant generation
# ──────────────────────────────────────────────────────────────────


def generate_variants(term: str) -> list[str]:
    """Generate STT-friendly variants for a jargon term.

    Applies four deterministic rules:
    1. CamelCase splitting:      SvelteKit -> "svelte kit"
    2. Lowercase flattening:     SvelteKit -> "sveltekit"
    3. Acronym letter-spacing:   TWAP -> "t w a p" (ALL_CAPS ≤5 chars only)
    4. Phonetic substitution:    PyTorch -> "pie torch"

    Returns deduplicated, sorted list excluding the canonical term itself.
    """
    canonical_lower = term.lower()
    variants: set[str] = set()

    parts = _split_camel_case(term)

    # Rule 1: CamelCase split → lowercase words
    if len(parts) > 1:
        variants.add(" ".join(p.lower() for p in parts))

    # Rule 2: Lowercase flattening (join all parts without space/underscore)
    flat = "".join(p.lower() for p in parts)
    variants.add(flat)

    # Rule 3: Acronym letter-spacing (ALL_CAPS terms ≤5 chars after stripping _)
    stripped = term.replace("_", "")
    if stripped.isupper() and len(stripped) <= 5:
        spaced = " ".join(ch.lower() for ch in stripped)
        variants.add(spaced)

    # Rule 4: Phonetic substitution on CamelCase parts
    variants.update(_expand_phonetic_variants(parts))

    # Remove the canonical form for single-part terms (e.g. "twap" for "TWAP").
    # For multi-part CamelCase terms (e.g. "pytorch" for "PyTorch"), keep the
    # flat lowercase form — it's a valid STT variant distinct from the canonical.
    if len(parts) == 1:
        variants.discard(canonical_lower)

    return sorted(variants)


# ──────────────────────────────────────────────────────────────────
# Claude Code command scanning
# ──────────────────────────────────────────────────────────────────


def _generate_command_variants(stem: str) -> list[str]:
    """Generate voice-friendly variants for a slash command stem.

    Strategy (addressing max_phrase_words=3 limit):
    - Single-word commands: "slash <word>" (2 words, fits in window)
    - Multi-word commands: both "slash <words>" AND bare "<words>" variants.
      Bare variants avoid exceeding the corrector's sliding window limit.

    Examples:
        "commit"         -> ["slash commit"]
        "review-pr"      -> ["review p r", "slash review p r", "slash review pr"]
        "commit-push-pr" -> ["commit push p r", "commit push pr",
                             "slash commit push p r", "slash commit push pr"]
    """
    parts = stem.split("-")
    parts = [p for p in parts if p]  # drop empty from leading/trailing hyphens
    if not parts:
        return []

    # Build base + phonetic variants from parts
    phonetic_combos = _expand_phonetic_variants(parts)
    variants: set[str] = set()

    for combo in phonetic_combos:
        # Always add "slash" prefixed variant
        variants.add("slash " + combo)
        # For multi-word commands (2+ parts), also add bare variant
        # to stay within max_phrase_words=3 corrector window
        if len(parts) >= 2:
            variants.add(combo)

    # Remove the bare stem itself (e.g. "commit" alone would cause false positives)
    base = " ".join(p.lower() for p in parts)
    if len(parts) == 1:
        variants.discard(base)

    return sorted(variants)


def _scan_claude_commands(root: Path) -> dict[str, list[str]]:
    """Scan .claude/commands/ directories for custom slash command names.

    Looks in:
      - root/.claude/commands/**/*.md (project-local commands, including subdirs)

    Subdirectory commands use colon notation: subdir/cmd.md -> /subdir:cmd

    Returns:
        dict mapping canonical form (e.g. "/review-pr") to list of voice variants.
    """
    cmd_dir = root / ".claude" / "commands"
    if not cmd_dir.is_dir():
        return {}

    commands: dict[str, list[str]] = {}
    for md_file in cmd_dir.rglob("*.md"):
        # Skip non-.md suffixes like .md.backup, .md.20250827_104606
        if md_file.suffix != ".md":
            continue
        # Skip README files
        if md_file.stem.lower() == "readme":
            continue
        # Skip hidden files and very short stems
        if md_file.stem.startswith(".") or len(md_file.stem) < 2:
            continue

        # Build canonical form from relative path
        rel = md_file.relative_to(cmd_dir)
        if len(rel.parts) == 1:
            # Direct child: /stem
            canonical = "/" + md_file.stem
            variants = _generate_command_variants(md_file.stem)
        else:
            # Subdirectory: /subdir:stem (colon notation)
            subdir_parts = list(rel.parts[:-1])
            canonical = "/" + ":".join(subdir_parts + [md_file.stem])
            # Variants use space-separated form: "slash subdir stem"
            all_parts = subdir_parts + [md_file.stem]
            combined_stem = "-".join(all_parts)
            variants = _generate_command_variants(combined_stem)

        if variants:
            commands[canonical] = variants

    logger.info("Found %d Claude commands in %s", len(commands), cmd_dir)
    return commands


def mine_claude_commands(
    scan_paths: list[Path],
    config: JargonConfig | None = None,
) -> dict[str, list[str]]:
    """Scan for Claude Code custom commands and return {canonical: [variants]}.

    Scans:
      - Each scan_path/.claude/commands/**/*.md
      - ~/.claude/commands/**/*.md (global user commands)
    """
    if config is None:
        config = JargonConfig()

    existing = _load_existing_terms(config)

    commands: dict[str, list[str]] = {}
    for path in scan_paths:
        if path.is_dir():
            commands.update(_scan_claude_commands(path))

    # Also scan ~/.claude/commands/ for global user commands
    home_claude_root = Path.home()
    if (home_claude_root / ".claude" / "commands").is_dir():
        logger.info("Also scanning ~/.claude/commands/ for global commands")
        commands.update(_scan_claude_commands(home_claude_root))

    # Filter out commands already known in existing jargon dicts
    commands = {k: v for k, v in commands.items() if k not in existing}

    return commands


def write_claude_commands_yaml(
    commands: dict[str, list[str]],
    output_path: Path,
) -> int:
    """Write discovered Claude commands to a YAML jargon file.

    Merges with existing file without overwriting hand-edits.
    Returns count of new terms written.
    """
    if not commands:
        return 0

    # Load existing file if present (for merge)
    existing_terms: dict[str, list[str]] = {}
    if output_path.exists():
        with open(output_path) as f:
            data = yaml.safe_load(f) or {}
        existing_terms = data.get("terms", {})

    # Only add new terms
    added = 0
    terms_to_write: dict[str, list[str]] = dict(existing_terms)
    for canonical, variants in commands.items():
        if canonical in terms_to_write:
            continue
        terms_to_write[canonical] = variants
        added += 1

    if added == 0:
        return 0

    # Write YAML with header comment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Auto-generated by Veery jargon miner (Claude Code commands)\n"
        f"# Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"# Edit freely — re-running --mine will merge without overwriting.\n\n"
    )
    yaml_body = yaml.dump(
        {"terms": terms_to_write},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    tmp_path = output_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        f.write(header)
        f.write(yaml_body)
    tmp_path.replace(output_path)

    logger.info("Wrote %d new Claude commands to %s", added, output_path)
    return added


# Directories to always skip
_SKIP_DIRS = frozenset({
    ".git", ".venv", "__pycache__", "node_modules", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs",
})

# Standard library module names to skip
_STDLIB_NAMES = frozenset({
    "os", "sys", "re", "json", "yaml", "math", "time", "datetime",
    "pathlib", "logging", "typing", "collections", "functools",
    "itertools", "abc", "enum", "dataclasses", "copy", "io",
    "subprocess", "threading", "multiprocessing", "unittest",
    "argparse", "configparser", "csv", "hashlib", "hmac",
    "http", "socket", "ssl", "urllib", "tempfile", "shutil",
    "glob", "fnmatch", "stat", "struct", "warnings",
})

_CAMEL_RE = re.compile(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$|^[A-Z][a-z]+(?:[A-Z]+[a-z]*)+$")
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")


def _is_interesting_name(name: str) -> bool:
    """Check if a name looks like domain jargon (CamelCase or ALL_CAPS)."""
    if len(name) < 3:
        return False
    if name.startswith(("_", "test", "Test")):
        return False
    if name in {"setUp", "tearDown", "setUpClass", "tearDownClass"}:
        return False
    # ALL_CAPS (likely acronym/constant): TWAP, VWAP, GMV
    if _ALL_CAPS_RE.match(name):
        return True
    # CamelCase with mixed case (likely proper noun): DuckDB, PyTorch
    if _CAMEL_RE.match(name):
        return True
    return False


def _extract_names_from_ast(source: str) -> list[str]:
    """Extract interesting identifiers from Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        # Class names
        if isinstance(node, ast.ClassDef):
            if _is_interesting_name(node.name):
                names.append(node.name)
        # Import names
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                real_name = alias.asname or alias.name
                if _is_interesting_name(real_name):
                    names.append(real_name)
        # ALL_CAPS assignments (top-level constants)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _ALL_CAPS_RE.match(target.id):
                    if len(target.id) >= 3:
                        names.append(target.id)

    return names


def _scan_directory(root: Path, max_files: int = 10_000) -> Counter[str]:
    """Walk a directory tree, parse Python files, count term occurrences."""
    counter: Counter[str] = Counter()
    file_count = 0

    for py_file in root.rglob("*.py"):
        # Skip excluded directories
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue

        file_count += 1
        if file_count > max_files:
            logger.warning("Hit file limit (%d), stopping scan", max_files)
            break

        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for name in _extract_names_from_ast(source):
            counter[name] += 1

    logger.info("Scanned %d Python files", min(file_count, max_files))
    return counter


def _load_existing_terms(config: JargonConfig) -> set[str]:
    """Load all canonical terms from existing YAML dictionaries."""
    existing: set[str] = set()
    for dict_path_str in config.dict_paths:
        dict_path = Path(dict_path_str)
        if not dict_path.is_absolute():
            dict_path = PROJECT_ROOT / dict_path
        if not dict_path.exists():
            continue
        with open(dict_path) as f:
            data = yaml.safe_load(f) or {}
        for canonical in data.get("terms", {}):
            existing.add(canonical)
    return existing


def mine_terms(
    scan_paths: list[Path],
    config: JargonConfig | None = None,
    max_files: int = 10_000,
) -> list[tuple[str, int, bool]]:
    """Scan paths for potential jargon terms.

    Returns:
        List of (term, frequency, already_known) sorted by frequency desc.
    """
    if config is None:
        config = JargonConfig()

    existing = _load_existing_terms(config)

    combined: Counter[str] = Counter()
    for path in scan_paths:
        if path.is_dir():
            combined += _scan_directory(path, max_files=max_files)
        elif path.suffix == ".py":
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                for name in _extract_names_from_ast(source):
                    combined[name] += 1
            except OSError:
                pass

    # Filter out stdlib names
    for name in list(combined):
        if name.lower() in _STDLIB_NAMES:
            del combined[name]

    results = [
        (term, count, term in existing)
        for term, count in combined.most_common()
    ]
    return results


def write_mined_yaml(
    results: list[tuple[str, int, bool]],
    output_path: Path,
    scan_paths: list[Path],
) -> int:
    """Write new mined terms with auto-generated variants to a YAML file.

    If the output file already exists, merges new terms without overwriting
    existing entries (safe to re-run without losing hand-edits).

    Returns:
        Count of new terms written.
    """
    # Filter to NEW terms only
    new_terms = [(term, freq) for term, freq, known in results if not known]
    if not new_terms:
        return 0

    # Load existing file if present (for merge)
    existing_terms: dict[str, list[str]] = {}
    if output_path.exists():
        with open(output_path) as f:
            data = yaml.safe_load(f) or {}
        existing_terms = data.get("terms", {})

    # Build new terms dict, skipping any already in the file
    added = 0
    terms_to_write: dict[str, list[str]] = dict(existing_terms)
    for term, _freq in new_terms:
        if term in terms_to_write:
            continue
        variants = generate_variants(term)
        terms_to_write[term] = variants
        added += 1

    if added == 0:
        return 0

    # Write YAML with header comment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths_str = ", ".join(str(p) for p in scan_paths)
    header = (
        f"# Auto-generated by Veery jargon miner\n"
        f"# Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"# Scanned: {paths_str}\n"
        f"# Edit freely — re-running --mine will merge without overwriting.\n\n"
    )
    yaml_body = yaml.dump(
        {"terms": terms_to_write},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    tmp_path = output_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        f.write(header)
        f.write(yaml_body)
    tmp_path.replace(output_path)

    logger.info("Wrote %d new terms to %s", added, output_path)
    return added


def print_mining_report(
    results: list[tuple[str, int, bool]],
    *,
    written_count: int = 0,
    output_path: Path | None = None,
) -> None:
    """Print a formatted report of mining results."""
    if not results:
        print("No potential jargon terms found.")
        return

    new_count = sum(1 for _, _, known in results if not known)
    known_count = sum(1 for _, _, known in results if known)

    print(f"\nFound {len(results)} potential jargon terms ({new_count} new, {known_count} already known):\n")

    for term, freq, known in results:
        status = "already known" if known else "NEW"
        print(f"  {term:<30} (seen {freq:>3} times) -- {status}")

    if written_count > 0 and output_path is not None:
        print(f"\nWrote {written_count} new terms to {output_path}")
        print("Terms will be active on next launch (jargon/mined.yaml is loaded by default).")
    elif new_count > 0:
        print("\nTo add terms, append them to jargon/quant_finance.yaml or jargon/tech.yaml")
        print("Format:")
        print("  TermName:")
        print("    - variant spelling 1")
        print("    - variant spelling 2")
