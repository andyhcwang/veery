"""Configuration dataclasses with sensible defaults. Optional YAML override via config.yaml."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Project root (two levels up from this file: src/veery/config.py → veery/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 32  # 32ms chunks → 512 samples at 16kHz (Silero VAD required size)
    wait_timeout_sec: float = 30.0  # Timeout for wait-based recording flows
    manual_max_duration_sec: float | None = None  # Optional hard cap for manual hold/toggle recording
    input_gain: float = 1.0  # Multiplier for input audio (increase for quiet microphones)

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def wait_timeout_samples(self) -> int:
        return int(self.sample_rate * self.wait_timeout_sec)


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.4  # Speech probability threshold (lower for accented speech)
    silence_duration_sec: float = 2.0  # Silence after speech → stop (longer for non-native pauses)
    min_speech_duration_sec: float = 0.3  # Minimum speech to count as valid


@dataclass(frozen=True)
class STTConfig:
    model_name: str = "iic/SenseVoiceSmall"
    language: str = "auto"  # Auto-detect for code-switching
    device: str = "cpu"  # CPU is more reliable than MPS on macOS
    backend: str = "whisper"  # "sensevoice" or "whisper"
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    whisper_use_jargon_prompt: bool = True
    whisper_prompt_terms_limit: int = 64
    whisper_prompt_char_limit: int = 400
    whisper_initial_prompt: str | None = None


STT_BACKENDS: tuple[tuple[str, str], ...] = (
    ("sensevoice", "SenseVoice-Small (Chinese-optimized)"),
    ("whisper", "Whisper Large-v3-turbo (multilingual-robust)"),
)


@dataclass(frozen=True)
class JargonConfig:
    dict_paths: tuple[str, ...] = (
        "jargon/quant_finance.yaml",
        "jargon/tech.yaml",
        "jargon/claude_code.yaml",
        "jargon/mined.yaml",
        "jargon/mined_commands.yaml",
    )
    learned_path: str | None = "jargon/learned.yaml"  # Tier 2: auto-learned terms
    fuzzy_threshold: int = 82  # rapidfuzz score_cutoff (0-100, tuned down from 85 for accented STT)
    max_phrase_words: int = 3  # Try 3-word, 2-word, 1-word phrases


@dataclass(frozen=True)
class LearningConfig:
    enabled: bool = True
    learned_path: str = "jargon/learned.yaml"
    promotion_threshold: int = 3
    correction_hotkey: str = "cmd+shift+r"


@dataclass(frozen=True)
class HotkeyConfig:
    key_combo: str = "right_cmd"  # Push-to-talk: hold to record, release to process
    mode: str = "hold"  # "hold" = push-to-talk, "toggle" = press-to-toggle


@dataclass(frozen=True)
class OutputConfig:
    cgevent_char_limit: int = 500  # Use CGEvent typing below this, clipboard above
    paste_delay_ms: int = 50  # Delay between clipboard write and Cmd+V


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    jargon: JargonConfig = field(default_factory=JargonConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)


def _filter_keys(cls: type, raw: dict) -> dict:
    """Keep only keys that match dataclass fields, warn on unknown ones."""
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, ", ".join(sorted(unknown)))
    return {k: v for k, v in raw.items() if k in valid}


def _normalize_audio_config(raw: dict) -> dict:
    """Map deprecated audio keys onto the current config contract."""
    audio_raw = dict(raw or {})
    legacy_wait_timeout = audio_raw.pop("max_duration_sec", None)
    if legacy_wait_timeout is None:
        return audio_raw

    if "wait_timeout_sec" in audio_raw:
        logger.warning(
            "audio.max_duration_sec is deprecated and ignored because audio.wait_timeout_sec is set.",
        )
        return audio_raw

    logger.warning("audio.max_duration_sec is deprecated; use audio.wait_timeout_sec instead.")
    audio_raw["wait_timeout_sec"] = legacy_wait_timeout
    return audio_raw


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config from YAML file, falling back to defaults for missing fields."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        return AppConfig()

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        audio_raw = _filter_keys(AudioConfig, _normalize_audio_config(raw.get("audio", {})))
        jargon_raw = _filter_keys(JargonConfig, raw.get("jargon", {}))
        if "dict_paths" in jargon_raw and isinstance(jargon_raw["dict_paths"], list):
            jargon_raw["dict_paths"] = tuple(jargon_raw["dict_paths"])

        cfg = AppConfig(
            audio=AudioConfig(**audio_raw),
            vad=VADConfig(**_filter_keys(VADConfig, raw.get("vad", {}))),
            stt=STTConfig(**_filter_keys(STTConfig, raw.get("stt", {}))),
            jargon=JargonConfig(**jargon_raw),
            hotkey=HotkeyConfig(**_filter_keys(HotkeyConfig, raw.get("hotkey", {}))),
            output=OutputConfig(**_filter_keys(OutputConfig, raw.get("output", {}))),
            learning=LearningConfig(**_filter_keys(LearningConfig, raw.get("learning", {}))),
        )

        # Validate audio config.
        audio_kwargs = dict(vars(cfg.audio))
        rebuild_audio = False

        wait_timeout = cfg.audio.wait_timeout_sec
        if not isinstance(wait_timeout, (int, float)) or wait_timeout <= 0:
            logger.warning(
                "wait_timeout_sec must be a positive number (got %r), resetting to 30.0",
                wait_timeout,
            )
            audio_kwargs["wait_timeout_sec"] = 30.0
            rebuild_audio = True

        manual_max_duration = cfg.audio.manual_max_duration_sec
        if manual_max_duration is not None and (
            not isinstance(manual_max_duration, (int, float)) or manual_max_duration <= 0
        ):
            logger.warning(
                "manual_max_duration_sec must be a positive number or null (got %r), disabling it",
                manual_max_duration,
            )
            audio_kwargs["manual_max_duration_sec"] = None
            rebuild_audio = True

        gain = cfg.audio.input_gain
        if not isinstance(gain, (int, float)) or gain <= 0:
            logger.warning("input_gain must be a positive number (got %r), resetting to 1.0", gain)
            audio_kwargs["input_gain"] = 1.0
            rebuild_audio = True
        elif gain > 100:
            logger.warning("input_gain %.1f is unusually high (max recommended: 20.0), audio may clip", gain)

        if rebuild_audio:
            cfg = AppConfig(
                audio=AudioConfig(**audio_kwargs),
                vad=cfg.vad,
                stt=cfg.stt,
                jargon=cfg.jargon,
                hotkey=cfg.hotkey,
                output=cfg.output,
                learning=cfg.learning,
            )

        return cfg
    except Exception:
        logger.warning("Failed to parse %s, using defaults", config_path, exc_info=True)
        return AppConfig()
