# Contributing to VoiceFlow

Thanks for your interest in making bilingual dictation better! VoiceFlow is built by and for bilingual Chinese/English tech professionals, and community contributions are what keep it sharp.

## Ways to Contribute

### 1. Submit Jargon Terms (Easiest, Highest Impact)

The jargon system is VoiceFlow's core differentiator. Every term you add helps everyone.

**To add terms to an existing pack:**

1. Find the right pack in `jargon/community/`:
   - `ai_ml.yaml` -- AI/ML model names, concepts, tools
   - `devops_cloud.yaml` -- Kubernetes, cloud services, CI/CD
   - `frontend.yaml` -- Frameworks, UI libraries, state management

2. Add your term with realistic STT variants:

```yaml
terms:
  YourTerm:
    - how stt hears it
    - another common mishearing
    - chinese transliteration if applicable
```

3. Open a PR.

**To create a new domain pack:**

1. Create `jargon/community/your_domain.yaml`
2. Follow this format:

```yaml
# Domain Name Community Jargon Pack
# Covers [description of what this pack includes]
# Last updated: YYYY-MM-DD

terms:
  TermOne:
    - variant one
    - variant two
  TermTwo:
    - variant one
    - variant two
```

3. Open a PR with a brief description of the domain.

**Variant quality guidelines:**
- Each variant should be a plausible STT misrecognition, not just a typo
- Test by actually speaking the term to SenseVoice or Whisper if possible
- For acronyms, include the space-separated letter version (e.g., "m c p" for MCP)
- For terms bilingual speakers use, include Chinese transliterations (e.g., 深度求索 for DeepSeek)
- 3-5 variants per term is ideal

### 2. Report STT Errors

If VoiceFlow consistently gets a term wrong:

1. Open an issue with:
   - What you said (in your natural mixed zh/en)
   - What VoiceFlow produced
   - What you expected
   - Which STT backend you're using (SenseVoice or Whisper)

2. Even better: include the fix as a jargon YAML entry in your issue.

### 3. Code Contributions

Bug fixes, features, and improvements are welcome. For larger changes, open an issue first to discuss the approach.

## Development Setup

```bash
git clone https://github.com/andyhcwang/voiceflow.git
cd voiceflow
uv sync
```

### Running Tests

```bash
uv run pytest tests/ -v          # all tests
uv run pytest tests/ -x          # stop on first failure
```

### Linting

```bash
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
```

### Code Style

- Python 3.13, ruff for linting (line-length 120)
- Type hints on all function signatures
- Frozen dataclasses for config (immutable after construction)
- Logging via `logging.getLogger(__name__)` -- no print statements
- Keep audio callback (`_audio_callback`) fast: no allocations, no I/O, no logging
- Mock heavy dependencies (ML models, sounddevice) in tests

### Architecture Overview

```
Audio → STT (SenseVoice / Whisper) → Jargon Correction → Filler Removal → Paste
```

Key files:
- `src/voiceflow/app.py` -- Main orchestrator, menubar, state machine
- `src/voiceflow/audio.py` -- Audio capture with Silero VAD
- `src/voiceflow/stt.py` -- STT backends (SenseVoice + Whisper)
- `src/voiceflow/jargon.py` -- Fuzzy + phonetic jargon matching
- `src/voiceflow/corrector.py` -- Correction pipeline
- `src/voiceflow/miner.py` -- Codebase jargon mining
- `src/voiceflow/learner.py` -- Auto-learning from corrections

## PR Guidelines

- Keep PRs focused -- one feature or fix per PR
- Add tests for new functionality
- Run `uv run ruff check src/ tests/` before submitting
- All existing tests should pass (`uv run pytest tests/ -v`)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
