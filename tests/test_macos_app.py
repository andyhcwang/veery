"""Tests for the py2app launcher."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_launcher_prefers_live_src_over_stale_build_lib(monkeypatch) -> None:
    """The dev .app must import live source, not py2app's stale build copy."""
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    build_lib = root / "build" / "bdist.macosx-11.0-arm64" / "lib"

    for module_name in list(sys.modules):
        if module_name == "macos_app" or module_name == "veery" or module_name.startswith("veery."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    remaining_path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() not in {root, src_dir, build_lib}
    ]
    monkeypatch.setattr(sys, "path", [str(root), str(build_lib), *remaining_path])

    importlib.import_module("macos_app")
    veery_app = importlib.import_module("veery.app")

    assert Path(veery_app.__file__).resolve().is_relative_to(src_dir)
