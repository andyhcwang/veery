# VoiceFlow

A macOS dictation app for bilingual (EN/ZH) professionals with domain jargon support.

## Target User

Chinese native speakers in English-speaking tech environments who:
- Mix English technical terms into Chinese conversation naturally ("Sharpe Ratio" not "夏普率")
- Switch between Slack (English) and WeChat (Chinese) constantly
- Don't want to switch keyboard language for dictation
- Need domain-specific jargon recognized correctly

## Architecture

```
Audio → STT (SenseVoice / Whisper) → Jargon Dictionary → Filler Removal → Paste
         ~100ms                       0ms, fuzzy match     0ms
```

**Total latency**: ~100-200ms | **Running cost**: $0/mo (fully local)

### Processing Pipeline

1. **STT**: SenseVoice-Small (Chinese-optimized) or Whisper Large-v3-turbo (accent-robust), switchable at runtime
2. **Jargon correction**: Phonetic fuzzy matching against user's custom YAML dictionaries (deterministic, instant)
3. **Filler removal**: Strips "um", "uh", "嗯", "额" and other filler words from both languages

## Features

- **Push-to-talk & toggle mode**: Hold right Cmd to record, or switch to press-to-toggle for longer dictation
- **Bilingual jargon preservation**: English technical terms (API, Kubernetes, etc.) stay in English even in Chinese text
- **Customizable jargon dictionaries**: Edit quant finance, tech, or learned terms via the menubar
- **Filler word removal**: Automatically strips filler words in both English and Chinese
- **Dual STT backends**: Switch between SenseVoice (Chinese-optimized) and Whisper (accent-robust) from the menubar
- **Auto-learning**: Re-dictate within 30s to teach VoiceFlow new corrections
- **Visual overlay**: Floating pill indicator shows recording/processing status
- **Session stats**: Menubar shows dictation count for the current session

## Status

Beta. See `docs/DESIGN.md` for detailed design decisions and `docs/COMPETITIVE_ANALYSIS.md` for market positioning.
