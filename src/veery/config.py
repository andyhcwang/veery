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
    # The prompt is prefill work on EVERY dictation (~100-300ms at 400 chars);
    # usage-ranked terms mean a smaller budget still covers active vocabulary.
    whisper_prompt_terms_limit: int = 32
    whisper_prompt_char_limit: int = 250
    whisper_initial_prompt: str | None = None
    # Watchdog: max seconds a dictation may spend in PROCESSING before the
    # app force-resets to IDLE (recovers from backend hangs).
    processing_timeout_sec: float = 120.0


STT_BACKENDS: tuple[tuple[str, str], ...] = (
    ("sensevoice", "SenseVoice-Small (Chinese-optimized)"),
    ("whisper", "Whisper Large-v3-turbo (multilingual-robust)"),
)


@dataclass(frozen=True)
class StreamingConfig:
    """Incremental transcription during hold-to-talk capture.

    Segments are cut at short VAD pauses and transcribed in the background
    while recording continues, so on key release only the residual tail
    needs decoding (whisper.cpp-stream / WhisperLive architecture).
    """

    enabled: bool = False
    # Mid-clip finalize pause — deliberately shorter than the recording-end
    # vad.silence_duration_sec (2.0s), which stays untouched. 1.0s (up from
    # the initial 0.7) cuts only at real clause breaks: hesitation pauses in
    # accented speech were producing mid-clause cuts (worse per-segment
    # context, spurious join commas).
    finalize_pause_sec: float = 1.0
    # Whisper accuracy degrades badly below ~1s; shorter runs of speech keep
    # accumulating into the next segment instead of being cut.
    min_segment_sec: float = 1.0
    # Force-cut a no-pause monologue so segments stay well-conditioned.
    max_segment_sec: float = 18.0
    # Audio prepended from before the cut so a straddling word isn't clipped.
    overlap_ms: int = 200
    # How long the release path waits for in-flight segment decodes; audio
    # not committed by then is simply covered by the tail transcription.
    drain_timeout_sec: float = 3.0
    # Tail of already-committed text passed as context to the next segment's
    # decode. DEFAULT OFF (0): real-world accented usage confirmed the
    # design caveat — prompt conditioning degrades accuracy and can trigger
    # echo artifacts. Opt in with e.g. 120 if your speech decodes cleanly.
    rolling_prompt_chars: int = 0


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
    # CGEvent typing below this length, instant clipboard paste above. Typing
    # long text is visibly slow in apps that process keystrokes one by one.
    cgevent_char_limit: int = 150
    paste_delay_ms: int = 50  # Post-Cmd+V wait before clipboard restore (values <150 clamped to 150; see output.py)


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
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


def _validate_numeric(
    value: object,
    *,
    name: str,
    default: float,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
    integer: bool = False,
) -> tuple[float, bool]:
    """Validate a numeric config value against optional bounds.

    Returns ``(value, changed)``: the original value when valid, otherwise the
    ``default`` with ``changed=True`` (and a warning logged). ``bool`` is never
    accepted as a number even though it subclasses ``int``.
    """
    if integer:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if ok and low is not None:
        ok = value >= low if low_inclusive else value > low  # type: ignore[operator]
    if ok and high is not None:
        ok = value <= high if high_inclusive else value < high  # type: ignore[operator]
    if ok:
        return value, False  # type: ignore[return-value]
    logger.warning("%s invalid (got %r), resetting to %r", name, value, default)
    return default, True


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
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        audio_raw = _filter_keys(AudioConfig, _normalize_audio_config(raw.get("audio", {})))
        jargon_raw = _filter_keys(JargonConfig, raw.get("jargon", {}))
        if "dict_paths" in jargon_raw and isinstance(jargon_raw["dict_paths"], list):
            jargon_raw["dict_paths"] = tuple(jargon_raw["dict_paths"])

        cfg = AppConfig(
            audio=AudioConfig(**audio_raw),
            vad=VADConfig(**_filter_keys(VADConfig, raw.get("vad", {}))),
            stt=STTConfig(**_filter_keys(STTConfig, raw.get("stt", {}))),
            streaming=StreamingConfig(**_filter_keys(StreamingConfig, raw.get("streaming", {}))),
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

        audio_cfg = AudioConfig(**audio_kwargs) if rebuild_audio else cfg.audio

        # Validate VAD config.
        vad_kwargs = dict(vars(cfg.vad))
        rebuild_vad = False
        threshold, changed = _validate_numeric(
            cfg.vad.threshold, name="vad.threshold", default=0.4,
            low=0.0, high=1.0, low_inclusive=False, high_inclusive=False,
        )
        if changed:
            vad_kwargs["threshold"] = threshold
            rebuild_vad = True
        silence, changed = _validate_numeric(
            cfg.vad.silence_duration_sec, name="vad.silence_duration_sec", default=2.0,
            low=0.0, low_inclusive=False,
        )
        if changed:
            vad_kwargs["silence_duration_sec"] = silence
            rebuild_vad = True
        min_speech, changed = _validate_numeric(
            cfg.vad.min_speech_duration_sec, name="vad.min_speech_duration_sec", default=0.3,
            low=0.0, low_inclusive=True,
        )
        if changed:
            vad_kwargs["min_speech_duration_sec"] = min_speech
            rebuild_vad = True
        vad_cfg = VADConfig(**vad_kwargs) if rebuild_vad else cfg.vad

        # Validate STT config.
        stt_kwargs = dict(vars(cfg.stt))
        rebuild_stt = False
        if cfg.stt.backend not in ("sensevoice", "whisper"):
            logger.warning(
                "stt.backend must be 'sensevoice' or 'whisper' (got %r), resetting to 'whisper'",
                cfg.stt.backend,
            )
            stt_kwargs["backend"] = "whisper"
            rebuild_stt = True
        timeout, changed = _validate_numeric(
            cfg.stt.processing_timeout_sec, name="stt.processing_timeout_sec", default=120.0,
            low=0.0, low_inclusive=False,
        )
        if changed:
            stt_kwargs["processing_timeout_sec"] = timeout
            rebuild_stt = True
        stt_cfg = STTConfig(**stt_kwargs) if rebuild_stt else cfg.stt

        # Validate jargon config.
        jargon_kwargs = dict(vars(cfg.jargon))
        rebuild_jargon = False
        fuzzy, changed = _validate_numeric(
            cfg.jargon.fuzzy_threshold, name="jargon.fuzzy_threshold", default=82,
            low=0, high=100, integer=True,
        )
        if changed:
            jargon_kwargs["fuzzy_threshold"] = int(fuzzy)
            rebuild_jargon = True
        jargon_cfg = JargonConfig(**jargon_kwargs) if rebuild_jargon else cfg.jargon

        # Validate hotkey config.
        hotkey_kwargs = dict(vars(cfg.hotkey))
        rebuild_hotkey = False
        valid_combos = (
            "right_cmd", "left_cmd", "right_alt", "left_alt",
            "right_shift", "left_shift", "right_ctrl", "left_ctrl",
        )
        if cfg.hotkey.key_combo not in valid_combos:
            logger.warning(
                "hotkey.key_combo must be one of %s (got %r), resetting to 'right_cmd'",
                "/".join(valid_combos), cfg.hotkey.key_combo,
            )
            hotkey_kwargs["key_combo"] = "right_cmd"
            rebuild_hotkey = True
        if cfg.hotkey.mode not in ("hold", "toggle"):
            logger.warning(
                "hotkey.mode must be 'hold' or 'toggle' (got %r), resetting to 'hold'",
                cfg.hotkey.mode,
            )
            hotkey_kwargs["mode"] = "hold"
            rebuild_hotkey = True
        hotkey_cfg = HotkeyConfig(**hotkey_kwargs) if rebuild_hotkey else cfg.hotkey

        # Validate streaming config.
        streaming_kwargs = dict(vars(cfg.streaming))
        rebuild_streaming = False
        if not isinstance(cfg.streaming.enabled, bool):
            logger.warning(
                "streaming.enabled must be a boolean (got %r), resetting to False",
                cfg.streaming.enabled,
            )
            streaming_kwargs["enabled"] = False
            rebuild_streaming = True
        for field_name, default, low in (
            ("finalize_pause_sec", 0.7, 0.0),
            ("min_segment_sec", 1.0, 0.0),
            ("max_segment_sec", 18.0, 0.0),
            ("drain_timeout_sec", 3.0, 0.0),
        ):
            value, changed = _validate_numeric(
                streaming_kwargs[field_name], name=f"streaming.{field_name}",
                default=default, low=low, low_inclusive=False,
            )
            if changed:
                streaming_kwargs[field_name] = value
                rebuild_streaming = True
        overlap, changed = _validate_numeric(
            streaming_kwargs["overlap_ms"], name="streaming.overlap_ms",
            default=200, low=0, integer=True,
        )
        if changed:
            streaming_kwargs["overlap_ms"] = int(overlap)
            rebuild_streaming = True
        rolling, changed = _validate_numeric(
            streaming_kwargs["rolling_prompt_chars"], name="streaming.rolling_prompt_chars",
            default=0, low=0, integer=True,
        )
        if changed:
            streaming_kwargs["rolling_prompt_chars"] = int(rolling)
            rebuild_streaming = True
        if streaming_kwargs["max_segment_sec"] <= streaming_kwargs["min_segment_sec"]:
            logger.warning(
                "streaming.max_segment_sec (%r) must exceed min_segment_sec (%r); resetting to 18.0",
                streaming_kwargs["max_segment_sec"], streaming_kwargs["min_segment_sec"],
            )
            streaming_kwargs["max_segment_sec"] = 18.0
            rebuild_streaming = True
        if streaming_kwargs["finalize_pause_sec"] >= vad_cfg.silence_duration_sec:
            logger.warning(
                "streaming.finalize_pause_sec (%r) must be shorter than "
                "vad.silence_duration_sec (%r); resetting to 0.7",
                streaming_kwargs["finalize_pause_sec"], vad_cfg.silence_duration_sec,
            )
            streaming_kwargs["finalize_pause_sec"] = 0.7
            rebuild_streaming = True
        streaming_cfg = StreamingConfig(**streaming_kwargs) if rebuild_streaming else cfg.streaming

        # Cross-check: learned terms are written by the learner but read by jargon.
        if cfg.learning.enabled and cfg.jargon.learned_path is not None:
            write_path = Path(cfg.learning.learned_path)
            if not write_path.is_absolute():
                write_path = PROJECT_ROOT / write_path
            read_path = Path(cfg.jargon.learned_path)
            if not read_path.is_absolute():
                read_path = PROJECT_ROOT / read_path
            if write_path.resolve() != read_path.resolve():
                logger.warning(
                    "learning.learned_path (%s) differs from jargon.learned_path (%s); "
                    "learned terms will be written to one path but read from another",
                    write_path,
                    read_path,
                )

        return AppConfig(
            audio=audio_cfg,
            vad=vad_cfg,
            stt=stt_cfg,
            streaming=streaming_cfg,
            jargon=jargon_cfg,
            hotkey=hotkey_cfg,
            output=cfg.output,
            learning=cfg.learning,
        )
    except Exception:
        logger.warning("Failed to parse %s, using defaults", config_path, exc_info=True)
        return AppConfig()
