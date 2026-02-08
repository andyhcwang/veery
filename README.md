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
Audio → SenseVoice-Small (234M, local) → Jargon Dictionary → Qwen2.5-3B (MLX, local) → Paste
         70ms/10s audio                   0ms, fuzzy match     100ms grammar fix
```

**Total latency**: ~150-300ms | **Running cost**: $0/mo (fully local)

### Three Processing Layers

1. **STT**: SenseVoice-Small — best code-switching model at 234M params, 15x faster than Whisper
2. **Jargon correction**: Phonetic fuzzy matching against user's custom dictionary (deterministic, instant)
3. **Grammar polish**: Qwen2.5-3B via MLX — fixes grammar, resolves ambiguity, $0 cost (local)

## Status

Early development. See `docs/DESIGN.md` for detailed design decisions.
