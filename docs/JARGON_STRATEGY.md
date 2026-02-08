# Jargon Freshness Strategy

*How VoiceFlow keeps its jargon dictionaries current with trending tech terminology.*

---

## Table of Contents

1. [The Problem: STT Training Data Cutoffs](#the-problem-stt-training-data-cutoffs)
2. [Community Jargon Pack System](#community-jargon-pack-system)
3. [Jargon Freshness Strategy](#jargon-freshness-strategy)
4. [Competitive Advantage Analysis](#competitive-advantage-analysis)
5. [Trending Terms Seed List (2024-2025)](#trending-terms-seed-list-2024-2025)
6. [Implementation Roadmap](#implementation-roadmap)

---

## The Problem: STT Training Data Cutoffs

Every speech-to-text model has a training data cutoff. SenseVoice-Small and Whisper-Large-v3 were trained on data through mid-2023 at latest. This means they have never seen:

- **Model names:** DeepSeek, Claude 3.5, Gemini, Llama 3, Qwen 2.5, Mistral, Phi-3
- **Products/frameworks:** Cursor, v0, Bolt, Devin, Windsurf, Claude Code
- **Concepts:** MCP (Model Context Protocol), agentic workflows, RAG pipelines, vibe coding
- **Companies:** Perplexity, Groq, Together AI, Fireworks AI

When a bilingual developer says "I'm using DeepSeek R1 with RAG," STT models produce garbage: "deep seek are one with rag" or worse. The jargon correction system is the **only reliable fix** that doesn't require retraining billion-parameter models.

This is VoiceFlow's core competitive moat: transparent, user-controlled, instantly-updatable jargon that stays ahead of any model's training data.

---

## Community Jargon Pack System

### Pack Structure

Jargon packs are domain-specific YAML files that follow the existing format. Each pack covers a coherent knowledge domain:

```
jargon/
  packs/
    ai-models.yaml         # LLMs, diffusion models, embeddings
    ai-infra.yaml           # MLOps, serving, training frameworks
    frontend.yaml            # React, Next.js, Svelte, CSS frameworks
    backend.yaml             # FastAPI, Django, databases, ORMs
    devops.yaml              # Docker, K8s, Terraform, CI/CD
    crypto-defi.yaml         # DeFi protocols, L2s, wallets
    quant-finance.yaml       # (existing, promoted from jargon/)
    apple-dev.yaml           # Swift, SwiftUI, Xcode, Apple Silicon
    cloud.yaml               # AWS, GCP, Azure services
    security.yaml            # Pen testing, CVEs, auth protocols
```

### Pack YAML Format

Each pack extends the existing format with metadata for versioning and freshness tracking:

```yaml
# AI Models & LLMs Jargon Pack
# Maintainer: @username
# Last reviewed: 2025-12-15

metadata:
  domain: ai-models
  version: "2025.12"       # YYYY.MM versioning
  description: "LLM and AI model names, architectures, and terminology"
  reviewed: "2025-12-15"   # Last human review date

terms:
  DeepSeek:
    - deep seek
    - deep sick
    - dee peek
  DeepSeek-R1:
    - deep seek are one
    - deep seek r one
    - deep sicker one
  Claude:
    - claud
    - cloud (when preceded by "anthropic" or "use")
  Gemini:
    - gem in I
    - gem any
    - jimmy (common STT error)
```

The `metadata` section is optional and ignored by the current `jargon.py` loader (it reads only `terms:`), making packs backward-compatible. Metadata is for tooling, freshness tracking, and community governance.

### How Community Members Contribute

**Contribution workflow via GitHub PRs:**

1. **Fork and add terms.** Contributors edit or create a pack YAML file, adding terms with phonetic variants they've encountered in real STT output.

2. **Variant quality bar.** Each variant must be a plausible STT misrecognition, not just a spelling variant. Guidelines:
   - Run the term through SenseVoice or Whisper and record what it produces
   - Include space-separated letter variants for acronyms (e.g., "m c p" for MCP)
   - Include CJK-adjacent errors for bilingual contexts

3. **PR review checklist:**
   - [ ] Term is a proper noun, acronym, or domain-specific word (not a common English word)
   - [ ] Variants are realistic STT errors (not just typos)
   - [ ] No duplicate terms across packs (CI check)
   - [ ] Fuzzy threshold won't cause false positives (variant isn't too close to a common word)

4. **CI validation.** A GitHub Action validates contributed YAML:
   - Valid YAML syntax
   - No duplicate canonical terms across all packs
   - No variants that would collide with existing terms at the configured fuzzy threshold
   - Each canonical term has at least one variant

### Pack Versioning

- Packs use calendar versioning: `YYYY.MM` (e.g., `2025.12`)
- Version bumps on any term addition, removal, or variant change
- The `reviewed` date tracks when a human last verified all terms are still relevant
- Stale packs (>6 months without review) get flagged in the README

### Configuration

Users select which packs to load in `config.yaml`:

```yaml
jargon:
  dict_paths:
    - jargon/packs/ai-models.yaml
    - jargon/packs/ai-infra.yaml
    - jargon/packs/frontend.yaml
    - jargon/packs/quant-finance.yaml
  # Or load all packs in a directory (future enhancement):
  # pack_dirs:
  #   - jargon/packs/
```

This leverages the existing `dict_paths` tuple in `JargonConfig` with no code changes required. Users only load the domains they care about, keeping the reverse index lean and avoiding cross-domain false positives.

---

## Jargon Freshness Strategy

### The Three-Layer Freshness Model

VoiceFlow's jargon stays current through three complementary mechanisms, from broadest to most personalized:

```
Layer 1: Community Packs     (updated monthly by community, domain-specific)
Layer 2: Codebase Mining      (on-demand, user's own codebase, `--mine`)
Layer 3: Auto-Learning        (continuous, from user corrections)
```

**Layer 1: Community Packs** cover industry-wide terminology. A developer working on AI agents gets the `ai-models` and `ai-infra` packs. Updated via GitHub PRs as new tools and frameworks emerge.

**Layer 2: Codebase Mining** (`voiceflow --mine ~/code`) fills the gap between community knowledge and individual projects. A developer working on a Rust project gets `tokio`, `serde`, `axum` in their dictionary automatically. This is VoiceFlow's most unique feature.

**Layer 3: Auto-Learning** (the existing `CorrectionLearner`) handles the long tail — terms too niche or too new for community packs, and project-specific terms the miner didn't catch. The promotion threshold (default: 3 corrections) prevents one-off mistakes from polluting the dictionary.

### Surfacing `--mine` as a Killer Differentiator

The `--mine` feature is currently buried as a CLI flag. It should be a first-class part of the user experience:

**Onboarding flow:**
1. First launch: "Point VoiceFlow at your codebase for instant jargon detection"
2. Menu bar item: "Mine Jargon from Project..." (opens directory picker)
3. Results shown in a summary: "Found 47 new terms from your codebase. Added to your personal dictionary."

**Positioning language:**
- "Your dictation app should know your codebase. VoiceFlow does."
- "Mine your own repo. Your IDE names, your API endpoints, your domain objects -- instantly recognized."
- "No other dictation tool does this. SuperWhisper asks you to type each term. Wispr Flow hopes its cloud model eventually learns. VoiceFlow mines your code in seconds."

**Technical enhancements to `--mine`:**
- Extend beyond Python: support JS/TS (`import` statements, class names), Rust (`use` declarations, struct names), Go (`import`, exported names)
- Mine `package.json`, `Cargo.toml`, `go.mod` for dependency names
- Mine README/docs for proper noun capitalization
- Generate STT variants automatically using phonetic rules (split CamelCase into words, expand acronyms)

### Process for Keeping Community Packs Updated

**Monthly update cycle:**

1. **Trend detection (automated).** A lightweight script (or manual scan) checks:
   - Hacker News front page terms (trending libraries, frameworks, companies)
   - GitHub trending repos (new project names)
   - Product Hunt launches (developer tools)
   - Major conference announcements (WWDC, Google I/O, NeurIPS, etc.)

2. **Community PR window.** First week of each month: maintainers review and merge community-submitted terms. Contributors are encouraged to submit terms they've personally encountered STT mangling.

3. **Variant validation.** Before merging, key variants are spot-checked against SenseVoice and Whisper to confirm they're realistic STT errors.

4. **Version bump and changelog.** Each pack version is bumped with a one-line changelog entry summarizing additions.

---

## Competitive Advantage Analysis

### Why Transparent YAML Beats Opaque Auto-Learn

| Dimension | VoiceFlow (YAML) | Wispr Flow (opaque) | SuperWhisper (replacements) |
|---|---|---|---|
| **Inspectable** | User can read/edit every term | Black box | Flat list, no fuzzy |
| **Portable** | Copy YAML to new machine, share with team | Locked to account | Export/import |
| **Version-controlled** | Git-tracked, diff-able, PR-reviewable | No history | No versioning |
| **Community-driven** | GitHub PRs, domain packs, shared across users | Per-user only | Third-party Trainer tool |
| **Debuggable** | See exactly why "deep seek" became "DeepSeek" | No visibility into corrections | Simple find-replace |
| **No false positives** | Tune fuzzy threshold, review variants, phonetic matching prevents collisions | Model decides, user can't override | Exact match only -- safe but limited |

**The key insight:** Wispr Flow and SuperWhisper treat vocabulary as a user-facing feature. VoiceFlow treats it as **infrastructure** -- versionable, shareable, composable. This is the difference between a personal preference and a professional tool.

### "Mine Your Own Codebase" as Unique Positioning

No competitor offers anything comparable:

- **Wispr Flow** relies on context awareness (reading screen text at runtime). This helps with terms visible on screen but misses the 90% of your codebase that isn't currently displayed.
- **SuperWhisper** has a "Trainer" tool where you manually type each term. For a codebase with 200+ domain-specific identifiers, this is impractical.
- **Every other tool** has no custom vocabulary at all.

VoiceFlow's miner walks your AST, extracts CamelCase class names, ALL_CAPS constants, import names, and builds a jargon dictionary in seconds. A developer switching to a new codebase runs `voiceflow --mine ~/new-project` and immediately gets accurate dictation for that project's terminology.

**This is a 10x improvement** over manually building a vocabulary list, and it's a feature that only makes sense for developer-focused dictation -- reinforcing VoiceFlow's positioning.

### Community-Driven Freshness vs. Stale LLM Training Data

The fundamental problem with relying on STT model training data for terminology:

1. **Training data lag.** Whisper-v3 was trained on data through ~2023. DeepSeek, Claude 3.5, Cursor, v0 -- none of these exist in its training data. Even if OpenAI releases Whisper-v4 tomorrow, it will still lag 6-12 months behind current terminology.

2. **The long tail.** Even current training data won't include niche but important terms: your company's internal tools, that new Rust crate with 500 GitHub stars, the acronym your team coined last month.

3. **Community velocity.** A community of 100 developers, each encountering STT failures in their daily work, can update jargon packs faster than any model retraining cycle. The feedback loop is: encounter error -> submit PR -> merged within days -> everyone benefits.

4. **Composability.** A quant finance developer loads `quant-finance.yaml` + `ai-models.yaml`. A frontend developer loads `frontend.yaml` + `ai-models.yaml`. Each gets a tailored dictionary without the noise of unrelated domains. LLM training can't offer this granularity.

---

## Trending Terms Seed List (2024-2025)

Terms that STT models consistently mangle, organized by domain. These should seed the initial community packs.

### AI Models & Companies

| Canonical | Common STT Errors | Priority |
|---|---|---|
| DeepSeek | deep seek, deep sick, dee peek | Critical |
| DeepSeek-R1 | deep seek are one, deep sicker one | Critical |
| DeepSeek-V3 | deep seek v3, deep seek vee three | Critical |
| Claude | claud, cloud | Critical |
| Claude Sonnet | cloud sonnet, claud sonnet | Critical |
| Claude Opus | cloud opus, claud opus | Critical |
| Gemini | gem in I, gem any, jimmy | Critical |
| Llama | llama (often OK), lama | Medium |
| Llama 3 | llama three, lama three | High |
| Qwen | chwen, queen, q wen | High |
| Mistral | mistral (often OK), miss trail | Medium |
| Phi-3 | fie three, phi three, fee three | Medium |
| Anthropic | anthropic (often OK), and thropic | High |
| Groq | grok, g rock, grow queue | High |
| Perplexity | perplexity (often OK) | Medium |
| Fireworks AI | fireworks a i | Low |
| Together AI | together a i | Low |
| Cohere | co here, co hear | Medium |
| xAI | x a i, ex ai | Medium |
| Grok | grok, grow | High |
| Stability AI | stability a i | Low |
| Midjourney | mid journey | Medium |
| DALL-E | dolly, doll e, dall e | High |
| Suno | sue no, soon oh | Medium |

### AI Concepts & Frameworks

| Canonical | Common STT Errors | Priority |
|---|---|---|
| RAG | rag (ambiguous with common word) | Critical |
| MCP | m c p, em see pee | Critical |
| agentic | a genetic, agented | High |
| LangChain | lang chain, language chain | High |
| LlamaIndex | llama index, lama index | High |
| LangGraph | lang graph, language graph | High |
| CrewAI | crew a i, crew ai | High |
| AutoGen | auto gen, autogen | Medium |
| LoRA | lora, lower a, low ra | High |
| QLoRA | q lora, q lower a | High |
| GGUF | g g u f, gee guff | High |
| GGML | g g m l, gee gee ml | Medium |
| MLX | m l x, ml x | High |
| vLLM | v l l m, vee llm | High |
| Ollama | oh llama, all llama | High |
| embeddings | embeddings (often OK) | Low |
| transformer | transformer (often OK) | Low |
| fine-tuning | fine tuning (often OK) | Low |
| prompt engineering | prompt engineering (often OK) | Low |
| token window | token window (often OK) | Low |
| context window | context window (often OK) | Low |
| hallucination | hallucination (often OK) | Low |
| guardrails | guardrails (often OK) | Low |

### Developer Tools (2024-2025)

| Canonical | Common STT Errors | Priority |
|---|---|---|
| Cursor | cursor (ambiguous with common word) | High |
| v0 | vee zero, v zero, v0 | Critical |
| Bolt | bolt (ambiguous) | Medium |
| Devin | devin (ambiguous with name) | Medium |
| Windsurf | wind surf | Medium |
| Claude Code | cloud code, claud code | High |
| Copilot | co pilot, copilot | Medium |
| Codeium | code e um, codium | Medium |
| Vercel | ver cell, ver sell | High |
| Supabase | super base, supa base, soup a base | High |
| PlanetScale | planet scale | Medium |
| Neon | neon (often OK) | Low |
| Turso | turso, tour so | Medium |
| Drizzle | drizzle (often OK) | Low |
| Prisma | prisma, prism a | Medium |
| tRPC | t r p c, tee rpc | High |
| Bun | bun (often OK, short) | Medium |
| Deno | deno, dee no | Medium |
| Astro | astro (often OK) | Low |
| SvelteKit | svelte kit | Medium |
| Tailwind | tail wind | Medium |
| shadcn | shad cn, shad see en | High |
| Zustand | zoo stand, zu stand | High |
| Zod | zod (often OK, short) | Medium |
| pnpm | p n p m, pee npm | High |
| Biome | biome (often OK) | Low |
| Oxlint | ox lint | Medium |

### Infrastructure & DevOps

| Canonical | Common STT Errors | Priority |
|---|---|---|
| Terraform | terraform (often OK) | Low |
| Pulumi | pull umi, poo loomy | High |
| Cloudflare Workers | cloud flare workers | Medium |
| Durable Objects | durable objects (often OK) | Low |
| Fly.io | fly i o, fly dot io | Medium |
| Railway | railway (often OK) | Low |
| Coolify | cool if I, cool ify | Medium |
| Kamal | camel, come all | High |
| Nix | nix (often OK, short) | Low |
| Podman | pod man | Medium |
| Argo CD | argo c d, argo see dee | Medium |
| Grafana | graph ana, grafana | Medium |

---

## Implementation Roadmap

### Phase 1: Seed Packs (Week 1)

- Create `jargon/packs/` directory structure
- Seed `ai-models.yaml` with the Critical and High priority terms from the seed list above
- Seed `ai-infra.yaml`, `frontend.yaml`, `devops.yaml` with initial terms
- Migrate existing `tech.yaml` and `quant_finance.yaml` into the packs structure (keep originals as aliases)
- Update default `dict_paths` in config to include packs

### Phase 2: Contribution Workflow (Week 2)

- Add `CONTRIBUTING.md` section on jargon pack contributions
- Create a PR template for jargon additions
- Add CI validation script (YAML lint, duplicate detection, collision check at fuzzy_threshold=82)
- Document the variant quality bar with examples

### Phase 3: Mine Enhancement (Week 3-4)

- Extend miner to support JS/TS, Rust, Go source files
- Add `package.json` / `Cargo.toml` / `go.mod` dependency name extraction
- Auto-generate phonetic variants from CamelCase splitting
- Add menu bar "Mine Jargon..." item that opens a directory picker

### Phase 4: Community Growth (Ongoing)

- Monthly update cycle for community packs
- Track which packs are most popular (download/config frequency)
- Feature "pack of the month" in release notes for trending domains
- Encourage contributors by crediting them in pack metadata

---

## Summary

VoiceFlow's jargon system is its strongest competitive moat. The combination of:

1. **Community packs** for industry-wide trending terms
2. **Codebase mining** for project-specific vocabulary
3. **Auto-learning** for the long tail of personal corrections

...creates a three-layer defense against STT training data staleness that no competitor can match. The transparent YAML format makes this system inspectable, portable, and community-driven -- turning a potential weakness (STT errors on new terms) into VoiceFlow's defining strength.
