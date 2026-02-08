"""Bootstrap jargon mining from Python source code."""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from pathlib import Path

import yaml

from voiceflow.config import PROJECT_ROOT, JargonConfig

logger = logging.getLogger(__name__)

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
    if name.startswith("_"):
        return False
    if name.startswith("test") or name.startswith("Test"):
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
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                real_name = alias.asname or alias.name
                if _is_interesting_name(real_name):
                    names.append(real_name)
        elif isinstance(node, ast.Import):
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


def print_mining_report(results: list[tuple[str, int, bool]]) -> None:
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

    if new_count > 0:
        print(f"\nTo add terms, append them to jargon/quant_finance.yaml or jargon/tech.yaml")
        print("Format:")
        print("  TermName:")
        print("    - variant spelling 1")
        print("    - variant spelling 2")
