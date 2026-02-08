# Veery

## Project Overview
macOS menubar dictation app for bilingual (Chinese/English) professionals.
Stack: SenseVoice-Small / Whisper (STT) + Silero VAD + rapidfuzz jargon + rumps (menubar).

## Build & Run
```bash
uv sync                          # install deps
uv run veery                 # launch menubar app
uv run veery --mine ~/code   # mine jargon from codebase
```

## Test
```bash
uv run pytest tests/ -v          # all tests
uv run pytest tests/ -x          # stop on first failure
uv run ruff check src/ tests/    # lint
```

## Architecture
- `app.py` — Main orchestrator, rumps menubar, state machine (IDLE → RECORDING → PROCESSING)
- `audio.py` — AudioRecorder with Silero VAD, sounddevice InputStream, push-to-talk / toggle modes
- `stt.py` — STT backends: SenseVoice-Small (FunASR) and Whisper (mlx-whisper)
- `corrector.py` — Pipeline: jargon → filler removal
- `jargon.py` — YAML-based fuzzy + phonetic jargon matching
- `config.py` — Frozen dataclasses, YAML override via `config.yaml`
- `overlay.py` — NSPanel-based floating pill overlay
- `output.py` — CGEvent typing (short text) or clipboard paste (long text)
- `learner.py` — Auto-learns corrections from re-dictation
- `miner.py` — Mines jargon from codebases

## Key Patterns
- All heavy models load in background thread (`_load_models`), `_ready` Event gates recording
- VAD is pre-loaded at startup for instant recording start
- `prepare_stream()` opens audio immediately on hotkey press for zero-latency capture
- Pre-speech buffer: 500ms ring buffer captures speech onset before VAD triggers
- Thread safety: `_state_lock` protects state machine transitions

## Code Style
- Python 3.13, ruff for linting
- Type hints on all function signatures
- Frozen dataclasses for config (immutable after construction)
- Logging via `logging.getLogger(__name__)` — no print statements
- Keep audio callback (`_audio_callback`) fast: no allocations, no I/O, no logging
- Mock heavy dependencies (ML models) in tests

## Known Lint Exceptions
- `overlay.py`: unused `objc` and `AppKit` imports (needed at runtime for PyObjC)
- `miner.py`: f-string without placeholders (line 189)
