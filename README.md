# VoiceFlow

**Dictation that speaks your language. Both of them.**

The only dictation app built for how bilingual Chinese/English tech professionals actually speak -- mixed-language, jargon-heavy, and running entirely on your Mac for free.

<!-- Demo GIF placeholder: Screen recording showing mixed zh/en dictation with jargon correction -->

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)](https://support.apple.com/en-us/116943)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes-brightgreen.svg)](https://github.com/andyhcwang/voiceflow)

## The Problem

You say "帮我review一下这个PR，看看API的latency有没有改善" and your dictation app produces garbage. Apple Dictation forces you to pick one language. Wispr Flow and SuperWhisper are English-first -- they don't understand that "Sharpe ratio" should stay in English when you're speaking Chinese, not become "夏普率". And every tool turns "PyTorch" into "pie torch" and "DuckDB" into "duck dee bee" because their models have never seen your jargon.

You've tried five dictation apps. They all fail at the same thing: **mixed-language technical speech**. You've resigned yourself to typing everything, even though dictation would be 3-5x faster.

VoiceFlow was built because no one else was going to build it.

## How It Works

```
Hold Right Cmd → Speak naturally in zh/en → Release → Text appears

Audio → STT → Jargon Correction → Filler Removal → Paste to active app
        ↓           ↓                    ↓
   SenseVoice    fuzzy+phonetic       strips "um",
   or Whisper    YAML matching        "嗯", "额"
```

- **Latency**: ~100-200ms end-to-end
- **Cost**: $0/month, forever
- **Privacy**: 100% local -- no audio data leaves your machine

## Features

- **Push-to-talk & toggle mode** -- Hold Right Cmd to record, or switch to press-to-toggle for longer dictation sessions
- **Bilingual jargon preservation** -- English terms stay in English in Chinese text ("API", "Kubernetes", "Sharpe ratio" never get transliterated)
- **Customizable YAML jargon dictionaries** -- fuzzy matching + phonetic matching catches STT errors like "pie torch" -> PyTorch, "duck dee bee" -> DuckDB
- **Community jargon packs** -- Pre-built packs for AI/ML, DevOps/Cloud, and Frontend/Web (contribute your own!)
- **Auto-learning from corrections** -- Re-dictate within 30 seconds and VoiceFlow learns the correction automatically
- **Codebase jargon mining** -- Run `--mine ~/code` to scan your projects and discover terms to add
- **Filler word removal** -- Strips "um", "uh", "嗯", "额", "那个" and other fillers in both languages
- **Dual STT backends** -- SenseVoice-Small (Chinese-optimized, fast) and Whisper Large-v3-turbo (accent-robust), switchable at runtime from the menubar
- **Visual overlay** -- Floating pill indicator shows recording/processing/success status
- **Fully local, fully open source** -- No cloud, no account, no telemetry. Read every line of code.

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/andyhcwang/voiceflow.git
cd voiceflow
bash install.sh   # checks prerequisites, installs deps
```

Or manually:

```bash
git clone https://github.com/andyhcwang/voiceflow.git
cd voiceflow
uv sync
```

### Run

```bash
uv run voiceflow
```

On first launch, VoiceFlow will:
1. Guide you through granting **Accessibility** and **Microphone** permissions
2. Download the STT model (~200MB for SenseVoice, ~1.5GB for Whisper) with a progress bar
3. Show a microphone icon in your menubar when ready

Hold **Right Cmd**, speak in whatever mix of Chinese and English comes naturally, release, and watch the text appear.

## The Jargon System

This is VoiceFlow's killer feature. STT models are trained on general speech -- they don't know your domain vocabulary. VoiceFlow fixes this with a three-layer correction system that runs in <1ms:

### 1. Exact matching

```yaml
# jargon/tech.yaml
terms:
  PyTorch:
    - pie torch
    - py torch
  DuckDB:
    - duck dee bee
    - duck DB
  Sharpe ratio:
    - sharp ratio
    - sharp issue
```

When the STT outputs "pie torch", VoiceFlow instantly corrects it to "PyTorch".

### 2. Fuzzy matching

Even if the STT output doesn't exactly match a variant, fuzzy matching (via [rapidfuzz](https://github.com/rapidfuzz/rapidfuzz)) catches close approximations. "pytorche" still matches "PyTorch". Threshold is tunable (default: 82/100).

### 3. Phonetic matching

For terms where the STT gets the sounds right but the spelling wrong, consonant-skeleton matching catches them. "NumPi" and "NumPy" share the same skeleton `nmp`, so VoiceFlow knows they're the same term.

### Adding your own terms

Create or edit a YAML file in `jargon/`:

```yaml
# jargon/my_domain.yaml
terms:
  MyCompanyName:
    - my company name
    - my company
  InternalTool:
    - internal tool
    - inter nal tool
```

Then add it to your `config.yaml`:

```yaml
jargon:
  dict_paths:
    - jargon/tech.yaml
    - jargon/quant_finance.yaml
    - jargon/my_domain.yaml
```

Or open jargon files directly from the VoiceFlow menubar under **Jargon Dictionaries**.

### Community jargon packs

Pre-built packs for common domains. To use them, add to your `config.yaml`:

```yaml
jargon:
  dict_paths:
    - jargon/tech.yaml
    - jargon/community/ai_ml.yaml         # LLM models, RAG, LoRA, etc.
    - jargon/community/devops_cloud.yaml   # Kubernetes, Terraform, CI/CD
    - jargon/community/frontend.yaml       # Next.js, React, Tailwind, etc.
```

Want to contribute a pack for your domain? See [Contributing](#contributing).

### Mining jargon from your codebase

VoiceFlow can scan your Python projects and suggest terms to add:

```bash
uv run voiceflow --mine ~/code ~/projects/ml-pipeline
```

It extracts CamelCase class names, ALL_CAPS constants, and imported module names, filters out standard library, and shows you what's new vs. already in your dictionaries.

### Auto-learning

When VoiceFlow gets a term wrong, just re-dictate the correction within 30 seconds. VoiceFlow detects the correction, logs it, and after 3 identical corrections, automatically promotes it to your learned dictionary (`jargon/learned.yaml`). No manual YAML editing needed.

## Configuration

VoiceFlow works out of the box with sensible defaults. To customize, create `config.yaml` in the project root:

```yaml
# config.yaml
stt:
  backend: sensevoice      # "sensevoice" or "whisper"

audio:
  max_duration_sec: 30.0    # Auto-stop after 30s

vad:
  threshold: 0.4            # Speech detection sensitivity (lower = more sensitive)
  silence_duration_sec: 2.0 # Seconds of silence before auto-stop

hotkey:
  mode: hold                # "hold" (push-to-talk) or "toggle" (press-to-toggle)

jargon:
  dict_paths:
    - jargon/tech.yaml
    - jargon/quant_finance.yaml
  fuzzy_threshold: 82       # Fuzzy match sensitivity (0-100)

output:
  cgevent_char_limit: 500   # Text shorter than this is typed character-by-character;
                            # longer text is pasted via clipboard

learning:
  enabled: true
  promotion_threshold: 3    # Corrections needed before auto-adding to dictionary
```

## Comparison

| | VoiceFlow | SuperWhisper | Wispr Flow | Apple Dictation |
|---|---|---|---|---|
| **Price** | Free forever | $8.49/mo | $10/mo | Free |
| **Privacy** | 100% local | Mostly local | Cloud-dependent | Cloud |
| **Bilingual zh/en** | Purpose-built | Multi-lang (generic) | Multi-lang (generic) | Single language only |
| **Jargon handling** | Fuzzy + phonetic + auto-learn | Find-and-replace | Cloud auto-learn | None |
| **Chinese STT** | SenseVoice (SOTA Chinese) | Whisper (English-first) | Proprietary | Apple ASR |
| **Custom dictionaries** | YAML (open, editable) | Vocabulary hints | Manual add | None |
| **Codebase mining** | Yes (`--mine`) | No | No | No |
| **Open source** | Yes | No | No | No |

VoiceFlow doesn't compete on polish or mobile support. It wins on an axis that matters to you: **it's the only tool that actually works for mixed zh/en technical speech**.

## Contributing

We welcome contributions, especially:

- **Jargon packs** -- Add terms for your domain (biotech, crypto, game dev, etc.)
- **Bug reports** -- Open an issue with your STT output and expected correction
- **STT improvements** -- New backend integrations, accuracy benchmarks

To submit a community jargon pack:

1. Create `jargon/community/your_domain.yaml` following the existing format
2. Include phonetic variants that STT models commonly produce
3. Open a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Credits

Built by [Andy Wang](https://x.com/AndyThinkMode).

**Core dependencies:**
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) -- Chinese-optimized STT by Alibaba DAMO Academy
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) -- Whisper on Apple Silicon via MLX
- [Silero VAD](https://github.com/snakers4/silero-vad) -- Voice activity detection
- [rapidfuzz](https://github.com/rapidfuzz/rapidfuzz) -- Fuzzy string matching
- [rumps](https://github.com/jaredks/rumps) -- macOS menubar framework

## License

MIT
