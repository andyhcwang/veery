# Design Decisions

## Why NOT a Cloud LLM for Post-Processing

| | Cloud LLM (Claude/GPT) | Local Qwen2.5-3B (MLX) |
|---|---|---|
| Latency | 200-2000ms (variable) | 50-150ms (consistent) |
| Cost at 500 dictations/day | $2.50-10/day | $0 |
| Offline | No | Yes |
| Privacy | Audio/text leaves device | Never leaves device |
| EN+ZH quality | Good | Good (Qwen = Alibaba, native bilingual) |
| Grammar correction | Excellent | Good enough for 1-2 sentence fixes |

The task is "fix grammar in 1-2 sentences" — not reasoning, not creative writing.
A 3B model is sufficient. Overkill models add latency and cost without proportional benefit.

## Why SenseVoice Over Whisper

- Chinese CER: 2.96% vs 5.14% (Whisper-Large-V3) — nearly 2x better
- Code-switching: Purpose-built vs emergent/unreliable
- Speed: 70ms/10s audio (15x faster than Whisper-Large)
- Size: 234M params vs 1.55B — runs lighter on Mac

## Why Phonetic Matching Before LLM

Most jargon errors are phonetically predictable:
- "sharp ratio" → Sharpe ratio (edit distance 1)
- "duck DB" → DuckDB (phonetic match)
- "tee wap" → TWAP (phonetic match)

A dictionary lookup at <1ms is infinitely cheaper and more reliable than asking
an LLM to guess. The LLM only handles what the dictionary can't:
grammar, ambiguity, context-dependent corrections.

## Layer 2 Model Selection Rationale

### Why Qwen2.5-3B specifically:
1. Alibaba model = best small model for Chinese text understanding
2. 3B is the sweet spot: 1.5B struggles with nuanced grammar, 7B is overkill + slower
3. MLX has first-class Qwen support (quantized 4-bit = ~2GB RAM)
4. Bilingual tokenizer — efficient for mixed EN/ZH text

### Upgrade path (optional, not default):
- Groq API with Llama 3.2 8B: 20-50ms, ~free ($0.0001/call)
- Only if user explicitly wants cloud-quality grammar correction

## Context-Aware Dictation (Future)

Detect active app → adjust behavior:
- **Slack/Email**: English grammar priority, professional tone
- **WeChat**: Chinese-first, casual tone, allow more code-switching
- **VS Code/Terminal**: Code-aware mode, recognize CLI commands and variable names
- **Browser**: Detect input language from page content
