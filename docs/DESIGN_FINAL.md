# VoiceFlow — Finalized Design (Post-Review)

## Design Decisions (from 3-agent review)

### Architecture
- **Threading**: Main (rumps) + Audio (sounddevice callback) + Worker (per-utterance daemon)
- **No asyncio** — fights rumps for main thread event loop
- **State transitions** must use `threading.Lock` (not just buffer lock)

### State Machine (expanded)
```
IDLE → RECORDING          (hotkey press)
RECORDING → PROCESSING    (silence detected after speech)
RECORDING → IDLE          (hotkey press = cancel, OR timeout with no speech)
PROCESSING → IDLE         (success: paste text)
PROCESSING → IDLE         (error: show notification)
```
- Hotkey during PROCESSING: **ignore** (simplest for v1)
- Max recording duration: 30s timeout → auto-stop

### ML Pipeline
| Component | Choice | RAM | Latency |
|-----------|--------|-----|---------|
| VAD | Silero VAD v5 (ONNX) | ~50MB | <1ms |
| STT | SenseVoice-Small via FunASR | ~500MB | 300-400ms (CPU) |
| Jargon | rapidfuzz + YAML dicts | ~10MB | <1ms |
| Grammar | **Qwen3-1.7B-Instruct-4bit** via MLX | ~1GB | 50-80ms |
| **Total** | | **~1.6GB** | **350-500ms** |

### Key Changes from Original Plan
1. **Grammar model**: Qwen2.5-3B → **Qwen3-1.7B-4bit** (saves 1GB, sufficient for grammar)
2. **Output**: pyperclip+pyautogui → **CGEvent typing** (no clipboard corruption)
3. **STT input**: Temp WAV file → **numpy array directly** to FunASR
4. **Model loading**: All eager → **VAD+STT eager, grammar lazy**
5. **Latency budget**: 200-350ms → **350-500ms** (realistic CPU inference)

### Output Mechanism (redesigned)
- **Short text (<500 chars)**: CGEvent typing via PyObjC — bypasses clipboard entirely
- **Long text (≥500 chars)**: NSPasteboard-aware clipboard paste (saves ALL pasteboard types, not just plain text)
- **No pyperclip, no pyautogui** — both replaced by PyObjC

### Permission Handling
1. Startup: Check `AXIsProcessTrusted()` via PyObjC
2. If false: Show notification + open System Settings Accessibility pane
3. Microphone: Handled automatically by sounddevice (system dialog)
4. Input Monitoring: Document in README (not detectable programmatically)
5. Add "Test Hotkey" menu item for self-diagnosis

### Fallback Chain
```
Full:     STT → Jargon → Grammar → Output
Degraded: STT → Jargon → Output          (if grammar model fails/not loaded)
Error:    Show notification                (if STT fails)
Fatal:    Show permission notification     (if audio/hotkey fails)
```

### Audio Lifecycle
- `InputStream.start()` on hotkey press (IDLE → RECORDING)
- `InputStream.stop()` on speech end or cancel (→ IDLE)
- Stream is NOT kept open while idle (saves battery, allows sleep)

### Dependencies (revised)
Core: sounddevice, numpy, torch, funasr, onnxruntime (silero-vad), mlx, mlx-lm
App: rumps, pynput, pyobjc-framework-Cocoa, pyobjc-framework-Quartz, pyyaml, rapidfuzz
Dev: pytest, ruff

Removed: pyperclip, pyautogui
Added: pyobjc-framework-Cocoa, pyobjc-framework-Quartz, onnxruntime
Changed: Qwen2.5-3B → Qwen3-1.7B

### Menubar Icons
- v1: Emoji (🎤, 🔴, ⏳) — works with rumps title
- v2: PNG template images (18x18 @1x, 36x36 @2x) for polish

### Config
- Hotkey configurable via config.yaml (default: Cmd+Shift+Space)
- Document bilingual keyboard conflict (Cmd+Shift+Space may be input source switch)
