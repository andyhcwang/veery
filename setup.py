"""Build an Apple Silicon Veery.app bundle with py2app."""

from __future__ import annotations

import sys
from pathlib import Path

from py2app.build_app import py2app as py2app_command
from setuptools import setup

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from veery import __version__  # noqa: E402

APP = ["macos_app.py"]
PYTHON_RESOURCES_DIR = f"lib/python{sys.version_info.major}.{sys.version_info.minor}"

DATA_FILES = [
    (PYTHON_RESOURCES_DIR, ["config.yaml"]),
    (
        f"{PYTHON_RESOURCES_DIR}/jargon",
        [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "jargon").glob("*.yaml"))],
    ),
    (
        f"{PYTHON_RESOURCES_DIR}/jargon/community",
        [
            path.relative_to(ROOT).as_posix()
            for path in sorted((ROOT / "jargon" / "community").glob("*.yaml"))
        ],
    ),
]

OPTIONS = {
    "arch": "arm64",
    "argv_emulation": False,
    "excludes": [
        "IPython",
        "llvmlite.tests",
        "numba.cuda.tests",
        "numba.tests",
        "numpy.testing",
        "numpy.tests",
        "pytest",
        "scipy.tests",
        "setuptools.tests",
        "sympy.plotting",
        "sympy.testing",
        "Tkinter",
        "_tkinter",
        "tkinter",
        "torch.testing",
        "torch.testing._internal",
    ],
    "includes": [
        "AppKit",
        "Cocoa",
        "Foundation",
        "Quartz",
        "PyObjCTools.AppHelper",
        "objc",
        "pynput.keyboard",
    ],
    "optimize": 0,
    "packages": [
        "veery",
        "funasr",
        "llvmlite",
        "mlx_whisper",
        "modelscope",
        "numba",
        "numpy",
        "onnxruntime",
        "pynput",
        "rapidfuzz",
        "rumps",
        "sounddevice",
        "soundfile",
        "tiktoken",
        "torch",
        "torchaudio",
        "yaml",
    ],
    "plist": {
        "CFBundleDisplayName": "Veery",
        "CFBundleIdentifier": "io.github.andyhcwang.veery",
        "CFBundleName": "Veery",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "14.0",
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": (
            "Veery needs microphone access to capture speech for on-device dictation."
        ),
    },
    "site_packages": True,
    "strip": False,
}


class VeeryPy2App(py2app_command):
    """Clear setuptools-style dependency metadata before py2app validates options."""

    def collect_recipedict(self):
        recipes = super().collect_recipedict()
        recipes.pop("tkinter", None)
        return recipes

    def finalize_options(self):
        self.distribution.install_requires = []
        super().finalize_options()


setup(
    app=APP,
    cmdclass={"py2app": VeeryPy2App},
    data_files=DATA_FILES,
    description="macOS bilingual dictation with domain jargon support",
    include_package_data=False,
    name="Veery",
    options={"py2app": OPTIONS},
    package_dir={"": "src"},
    packages=["veery"],
    version=__version__,
)
