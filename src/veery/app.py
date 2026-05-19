"""Veery menubar application. Orchestrates audio capture, STT, and text correction."""

from __future__ import annotations

import enum
import functools
import logging
import subprocess
import threading
import time
import webbrowser
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import rumps

from veery import __version__, sounds
from veery.audio import AudioRecorder, AudioSegment, StopReason
from veery.config import PROJECT_ROOT, STT_BACKENDS, AppConfig, load_config
from veery.corrector import TextCorrector
from veery.jargon import JargonCorrector
from veery.learner import CorrectionLearner
from veery.output import paste_to_active_app
from veery.overlay import (
    DownloadProgressOverlay,
    OverlayIndicator,
    PermissionGuideOverlay,
    check_permissions_granted,
)
from veery.stt import (
    SenseVoiceSTT,
    WhisperSTT,
    _is_model_cached,
    _is_sensevoice_cached,
    create_stt,
    ensure_model_downloaded,
    ensure_sensevoice_downloaded,
)

logger = logging.getLogger(__name__)

def _run_on_main_thread(fn: Callable[[], None]) -> None:
    """Schedule *fn* to run on the main (AppKit) thread.

    Uses PyObjCTools.AppHelper.callAfter (same approach as overlay.py) so
    rumps/AppKit UI mutations are always executed on the main run-loop.
    Falls back to direct execution on the main thread or when PyObjC is unavailable.
    """
    if threading.current_thread() is threading.main_thread():
        fn()
        return
    try:
        from PyObjCTools.AppHelper import callAfter

        callAfter(fn)
    except ImportError:
        fn()
    except Exception:
        logger.exception("Failed to dispatch to main thread")

_MODIFIER_KEYS = {
    "Key.cmd",
    "Key.cmd_l",
    "Key.cmd_r",
    "Key.alt",
    "Key.alt_l",
    "Key.alt_r",
    "Key.shift",
    "Key.shift_l",
    "Key.shift_r",
}
_CMD_MODIFIERS = {"Key.cmd", "Key.cmd_l", "Key.cmd_r"}
_ALT_MODIFIERS = {"Key.alt", "Key.alt_l", "Key.alt_r"}

# Maps config key_combo strings to pynput Key attributes and display labels.
_KEY_COMBO_MAP = {
    "right_cmd": ("cmd_r", "Right \u2318"),
    "left_cmd": ("cmd_l", "Left \u2318"),
    "right_alt": ("alt_r", "Right \u2325"),
    "left_alt": ("alt_l", "Left \u2325"),
    "right_shift": ("shift_r", "Right \u21e7"),
    "left_shift": ("shift_l", "Left \u21e7"),
    "right_ctrl": ("ctrl_r", "Right \u2303"),
    "left_ctrl": ("ctrl_l", "Left \u2303"),
}


@dataclass
class _ManualEditSession:
    original_text: str
    text: str
    cursor: int
    started_at: float
    last_event_at: float
    edit_count: int = 0


def _is_repetitive_hallucination(text: str) -> bool:
    """Detect repetitive Whisper hallucination (e.g. 'Why Why Why...').

    Returns True when a single word accounts for >80% of a 6+ word sequence.
    """
    words = text.split()
    if len(words) < 6:
        return False
    most_common_count = max(Counter(w.lower() for w in words).values())
    return most_common_count / len(words) > 0.8


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class VeeryApp(rumps.App):
    """macOS menubar app for bilingual dictation with jargon correction."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()

        super().__init__(
            name="Veery",
            title="\U0001f3a4",  # microphone emoji
            quit_button=None,  # We add our own quit to control ordering
        )

        # State machine
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._recording_finalizing = False
        self._recording_session_id = 0

        # Menu items
        self._status_item = rumps.MenuItem(f"Veery v{__version__}", callback=None)
        self._status_item.set_callback(None)

        # Status detail (shows model state or last action)
        self._detail_item = rumps.MenuItem("Loading models...", callback=None)
        self._detail_item.set_callback(None)

        # Recording mode: "hold" (push-to-talk) or "toggle" (press-to-toggle)
        self._recording_mode: str = self._config.hotkey.mode

        # Resolve hotkey label from config
        combo = self._config.hotkey.key_combo
        _, self._hotkey_label = _KEY_COMBO_MAP.get(combo, ("cmd_r", "Right \u2318"))

        # Hotkey info (updates based on mode)
        self._hotkey_item = rumps.MenuItem(self._hotkey_hint(), callback=None)
        self._hotkey_item.set_callback(None)

        # Recording mode toggle
        self._mode_item = rumps.MenuItem(
            self._mode_label(),
            callback=self._on_toggle_mode,
        )

        # STT backend selector submenu
        self._stt_backend: str = self._config.stt.backend
        self._stt_model_menu = rumps.MenuItem("STT Model")
        self._stt_menu_items: dict[str, rumps.MenuItem] = {}
        self._build_stt_model_submenu()

        # Jargon submenu
        self._edit_jargon_menu = rumps.MenuItem("Jargon Dictionaries")
        self._build_jargon_submenu()

        self.menu = [
            self._status_item,
            self._detail_item,
            None,  # separator
            self._hotkey_item,
            self._mode_item,
            self._stt_model_menu,
            None,  # separator
            self._edit_jargon_menu,
            None,  # separator
            rumps.MenuItem("About Veery", callback=self._on_about),
            None,  # separator
            rumps.MenuItem("Quit Veery", callback=self._on_quit),
        ]

        # Track stats
        self._session_count: int = 0

        # Components (initialized below)
        self._recorder: AudioRecorder | None = None
        self._stt: SenseVoiceSTT | WhisperSTT | None = None
        self._corrector: TextCorrector | None = None
        self._learner: CorrectionLearner | None = None
        self._hotkey_listener = None
        self._overlay = OverlayIndicator()
        self._pending_stt_cleanup: list[SenseVoiceSTT | WhisperSTT] = []
        self._stt_cleanup_lock = threading.Lock()

        # Last pasted text + timestamp for auto-correction detection
        self._last_pasted_text: str | None = None
        self._last_pasted_time: float = 0.0
        self._correction_window_sec: float = 30.0  # auto-learn if re-dictated within 30s
        self._manual_edit_window_sec: float = 45.0  # track manual edits after paste
        self._manual_edit_idle_sec: float = 1.5  # finalize when no edit keys for this long
        self._manual_edit_session: _ManualEditSession | None = None
        self._manual_edit_lock = threading.Lock()
        self._modifier_keys_down: set[str] = set()

        # Signals that recording stream is actually open and capturing audio
        self._recording_started = threading.Event()

        # Signals that all models are loaded and app is ready
        self._ready = threading.Event()

        # Progressive loading: tracks background Whisper download
        self._whisper_loading: bool = False
        self._whisper_download_cancelled: bool = False
        self._stt_switching: bool = False

        self._permission_overlay = PermissionGuideOverlay()

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Lightweight init — create objects without loading heavy models."""
        logger.info("Initializing Veery components...")

        # 1. Audio recorder (VAD loaded eagerly in _load_models)
        try:
            self._recorder = AudioRecorder(self._config.audio, self._config.vad)
            logger.info("AudioRecorder initialized")
        except Exception:
            logger.exception("Failed to initialize AudioRecorder")

        # 2. Jargon + corrector (lightweight, no model download)
        try:
            jargon = JargonCorrector(self._config.jargon)
            self._corrector = TextCorrector(jargon)
            logger.info("TextCorrector initialized")
        except Exception:
            logger.exception("Failed to initialize TextCorrector")

        # 3. Correction learner
        if self._config.learning.enabled:
            try:
                self._learner = CorrectionLearner(self._config.learning)
                logger.info("CorrectionLearner initialized")
            except Exception:
                logger.exception("Failed to initialize CorrectionLearner")

        # 4. Global hotkey listener
        self._start_hotkey_listener()

        # 5. Permission check — show guide overlay or proceed to model loading
        if check_permissions_granted():
            logger.info("All permissions granted, loading models...")
            threading.Thread(target=self._load_models, daemon=True).start()
        else:
            logger.info("Missing permissions, showing permission guide...")
            self._detail_item.title = "Waiting for permissions..."
            self._permission_overlay.show(on_complete=self._on_permissions_granted)

        logger.info("Veery initialized")

    def _set_detail(self, text: str) -> None:
        """Update the detail menu item title on the main thread."""
        _run_on_main_thread(lambda t=text: setattr(self._detail_item, "title", t))

    def _on_permissions_granted(self) -> None:
        """Called when all permissions are granted (from PermissionGuideOverlay)."""
        logger.info("Permissions granted, starting model loading...")
        self._set_detail("Loading models...")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _download_sensevoice_with_overlay(
        self,
        model_name: str,
    ) -> DownloadProgressOverlay | None:
        """Download SenseVoice if not cached, showing a progress overlay.

        Returns the overlay (still visible) so the caller can hide it,
        or None if no download was needed.
        """
        if _is_sensevoice_cached(model_name):
            self._set_detail("Loading SenseVoice...")
            return None

        overlay = DownloadProgressOverlay()
        overlay.show()
        self._set_detail("Downloading SenseVoice...")

        def _on_progress(fraction: float, detail: str) -> None:
            overlay.set_progress(fraction, detail)

        try:
            ensure_sensevoice_downloaded(model_name, progress_callback=_on_progress)
        except Exception:
            overlay.hide()
            raise
        overlay.set_progress(1.0, "Loading model into memory...")
        return overlay

    def _load_models(self) -> None:
        """Background thread: download/load heavy models with menubar status.

        Progressive loading strategy for first-run UX:
        - If Whisper is cached: load it directly (fast path, ~15s)
        - If Whisper is NOT cached and backend is "whisper": load SenseVoice first
          so user can dictate immediately, then download Whisper in the background
        - If backend is "sensevoice": load SenseVoice directly

        Shows a DownloadProgressOverlay when models need to be downloaded for the
        first time. If models are already cached, skips the overlay entirely.
        """
        download_overlay: DownloadProgressOverlay | None = None
        try:
            # VAD model (small, ~50MB — load first so recording is instant)
            _run_on_main_thread(lambda: setattr(self, "title", "\u2b07"))
            if self._recorder is not None:
                self._set_detail("Loading VAD model...")
                self._recorder._ensure_vad_loaded()
                logger.info("VAD model pre-loaded")

            # Pre-load audio feedback sounds so first recording has no delay
            sounds.preload()

            stt_cfg = self._config.stt
            whisper_cached = _is_model_cached(stt_cfg.whisper_model)

            if stt_cfg.backend == "whisper" and whisper_cached:
                # Fast path: Whisper already cached, load directly
                self._set_detail("Loading Whisper...")
                self._stt = create_stt(stt_cfg)
                self._apply_stt_runtime_hints(self._stt)
                logger.info("STT loaded (backend=whisper, cached)")

            elif stt_cfg.backend == "whisper" and not whisper_cached:
                # Progressive path: load SenseVoice first, then Whisper in background
                from dataclasses import replace

                download_overlay = self._download_sensevoice_with_overlay(stt_cfg.model_name)

                self._set_detail("Loading SenseVoice...")
                sv_config = replace(stt_cfg, backend="sensevoice")
                self._stt = create_stt(sv_config)
                self._apply_stt_runtime_hints(self._stt)
                logger.info("STT loaded (backend=sensevoice, progressive)")

                # Hide download overlay before marking ready
                if download_overlay is not None:
                    download_overlay.hide()
                    download_overlay = None

                # User can dictate now
                def _ui_progressive_ready():
                    self.title = "\U0001f3a4"
                    self._detail_item.title = "Ready"
                    self._update_stt_checkmarks("sensevoice")

                _run_on_main_thread(_ui_progressive_ready)
                self._ready.set()
                logger.info("STT ready — accepting dictation (SenseVoice while Whisper downloads)")

                # Spawn background Whisper download
                threading.Thread(target=self._load_whisper_background, daemon=True).start()
                return

            else:
                # SenseVoice backend: download with overlay if needed, then load
                download_overlay = self._download_sensevoice_with_overlay(stt_cfg.model_name)
                self._stt = create_stt(stt_cfg)
                self._apply_stt_runtime_hints(self._stt)
                logger.info("STT loaded (backend=sensevoice)")

            # Hide download overlay if it was shown
            if download_overlay is not None:
                download_overlay.hide()

            # Ready for recording
            def _ui_ready():
                self.title = "\U0001f3a4"
                self._detail_item.title = "Ready"

            _run_on_main_thread(_ui_ready)
            self._ready.set()
            logger.info("STT ready — accepting dictation")
        except Exception:
            logger.exception("Failed to load models")
            if download_overlay is not None:
                download_overlay.hide()

            def _ui_failed():
                self.title = "\u26a0"  # warning
                self._detail_item.title = "Model load failed"

            _run_on_main_thread(_ui_failed)
            self._ready.set()  # unblock anyway

    def _load_whisper_background(self) -> None:
        """Background thread: download Whisper and auto-switch when ready.

        Shows progress only in the menubar detail item (no overlay).
        All UI mutations are dispatched to the main thread via _run_on_main_thread.
        """
        stt_cfg = self._config.stt
        self._whisper_loading = True
        try:
            def _progress(fraction: float, detail: str) -> None:
                if "stalled" in detail.lower() or "retrying" in detail.lower():
                    title = f"Ready ({detail})"
                else:
                    pct = int(fraction * 100)
                    title = f"Ready (Whisper: {pct}%)"
                self._set_detail(title)

            ensure_model_downloaded(stt_cfg.whisper_model, progress_callback=_progress)
            self._set_detail("Ready (loading Whisper...)")

            # Load and warm up Whisper
            new_stt = create_stt(stt_cfg)
            self._apply_stt_runtime_hints(new_stt)

            if not self._whisper_download_cancelled:
                old_stt = self._stt
                self._stt = new_stt
                self._queue_stt_cleanup(old_stt)

                def _ui_switched():
                    self._update_stt_checkmarks("whisper")
                    self._detail_item.title = "Ready"

                _run_on_main_thread(_ui_switched)
                logger.info("Auto-switched to Whisper backend")
            else:
                self._queue_stt_cleanup(new_stt)
                logger.info("Whisper downloaded but user switched backend, skipping auto-switch")
                self._set_detail("Ready")
        except Exception:
            logger.exception("Background Whisper download/load failed")
            self._set_detail("Ready (Whisper failed)")
        finally:
            self._whisper_loading = False

    def _build_whisper_jargon_prompt(self) -> str | None:
        """Build a compact prompt that biases Whisper toward repo jargon spellings."""
        if not self._config.stt.whisper_use_jargon_prompt or self._corrector is None:
            return None

        terms = self._corrector.jargon.dictionary.canonical_terms
        if not terms:
            return None

        term_limit = max(0, self._config.stt.whisper_prompt_terms_limit)
        char_limit = max(0, self._config.stt.whisper_prompt_char_limit)
        if term_limit == 0 or char_limit == 0:
            return None

        prefix = "Technical dictation. Prefer these exact spellings: "
        suffix = "."
        if len(prefix) + len(suffix) >= char_limit:
            return None

        selected: list[str] = []
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            candidate = prefix + ", ".join(selected + [cleaned]) + suffix
            if len(candidate) > char_limit:
                break
            selected.append(cleaned)
            if len(selected) >= term_limit:
                break

        if not selected:
            return None
        return prefix + ", ".join(selected) + suffix

    def _apply_stt_runtime_hints(self, stt: SenseVoiceSTT | WhisperSTT | None) -> None:
        """Push runtime hinting into the active Whisper backend."""
        if not isinstance(stt, WhisperSTT):
            return
        try:
            stt.set_runtime_hints(prompt=self._build_whisper_jargon_prompt())
        except Exception:
            logger.exception("Failed to apply Whisper runtime hints")

    def _queue_stt_cleanup(self, stt: SenseVoiceSTT | WhisperSTT | None) -> None:
        """Release superseded STT backends when it is safe to do so."""
        if stt is None:
            return
        with self._stt_cleanup_lock:
            self._pending_stt_cleanup.append(stt)
        self._cleanup_pending_stt_async()

    def _cleanup_pending_stt_async(self) -> None:
        with self._state_lock:
            if self._state != State.IDLE:
                return
        threading.Thread(target=self._cleanup_pending_stt_resources, daemon=True).start()

    def _cleanup_pending_stt_resources(self) -> None:
        with self._stt_cleanup_lock:
            pending = list(self._pending_stt_cleanup)
            self._pending_stt_cleanup.clear()

        for stt in pending:
            release = getattr(stt, "release_resources", None)
            if not callable(release):
                continue
            try:
                release()
            except Exception:
                logger.exception("Failed to release superseded STT backend")

    def _update_stt_checkmarks(self, backend_id: str) -> None:
        """Update STT menu checkmarks and internal backend state."""
        self._stt_backend = backend_id
        for bid, item in self._stt_menu_items.items():
            item.state = 1 if bid == backend_id else 0

    def _start_hotkey_listener(self) -> None:
        """Start hotkey listener using the configured key_combo."""
        try:
            from pynput.keyboard import Key, Listener

            combo = self._config.hotkey.key_combo
            pynput_attr, _ = _KEY_COMBO_MAP.get(combo, ("cmd_r", "Right \u2318"))
            target_key = getattr(Key, pynput_attr, Key.cmd_r)

            logger.info("Registering push-to-talk key: %s (%s)", combo, self._hotkey_label)

            def on_press(key):
                self._on_global_key_press(key)
                if key == target_key:
                    self._on_key_down()

            def on_release(key):
                self._on_global_key_release(key)
                if key == target_key:
                    self._on_key_up()

            self._hotkey_listener = Listener(on_press=on_press, on_release=on_release)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()

            # pynput silently fails when Input Monitoring is denied:
            # CGEventTapCreate returns None and the thread exits without raising.
            # Give the thread a moment to attempt tap creation, then check.
            time.sleep(0.3)
            if not self._hotkey_listener.is_alive():
                logger.error("Hotkey listener died — Input Monitoring likely denied")
                self._set_detail("Hotkey failed — grant Input Monitoring")
                try:
                    rumps.notification(
                        "Veery",
                        "Hotkey listener failed",
                        "Grant Input Monitoring in System Settings "
                        "\u2192 Privacy & Security \u2192 Input Monitoring, then restart.",
                    )
                except Exception:
                    logger.warning("Could not send notification for hotkey failure")
                return
        except Exception:
            logger.exception("Failed to start hotkey listener")
            self._set_detail("Hotkey failed — check Input Monitoring")
            try:
                rumps.notification(
                    "Veery",
                    "Hotkey listener failed",
                    "Grant Input Monitoring in System Settings "
                    "\u2192 Privacy & Security \u2192 Input Monitoring, then restart.",
                )
            except Exception:
                logger.warning("Could not send notification for hotkey failure")

    # ------------------------------------------------------------------
    # Global key monitoring (manual-edit auto-learn)
    # ------------------------------------------------------------------

    def _normalize_key_event(self, key) -> tuple[str, str | None]:
        """Normalize pynput key objects into ('char'|'key', value)."""
        if key is None:
            return "key", None
        if isinstance(key, str):
            if len(key) == 1:
                return "char", key
            return "key", key
        char = getattr(key, "char", None)
        if isinstance(char, str) and len(char) == 1:
            return "char", char
        return "key", str(key)

    def _on_global_key_press(self, key) -> None:
        """Track global key presses to infer manual post-paste corrections."""
        key_kind, key_value = self._normalize_key_event(key)
        if key_value is None:
            return

        if key_kind == "key" and key_value in _MODIFIER_KEYS:
            with self._manual_edit_lock:
                self._modifier_keys_down.add(key_value)
            return

        should_finalize = False
        should_discard = False
        now = time.monotonic()

        with self._manual_edit_lock:
            session = self._manual_edit_session
            if session is None:
                return

            if now - session.started_at > self._manual_edit_window_sec:
                should_discard = True
            elif session.edit_count > 0 and now - session.last_event_at > self._manual_edit_idle_sec:
                should_finalize = True
            else:
                status = self._apply_manual_edit_key_locked(session, key_kind, key_value)
                if status == "discard":
                    should_discard = True
                elif status == "tracked":
                    session.last_event_at = now

        if should_discard:
            self._discard_manual_edit_monitor()
            return
        if should_finalize:
            self._finalize_manual_edit_learning()
            return

    def _on_global_key_release(self, key) -> None:
        """Track key releases so modifier state stays accurate."""
        key_kind, key_value = self._normalize_key_event(key)
        if key_kind == "key" and key_value in _MODIFIER_KEYS:
            with self._manual_edit_lock:
                self._modifier_keys_down.discard(key_value)

    def _begin_manual_edit_monitor(self, original_text: str) -> None:
        """Start a short-lived edit tracking session for recently inserted text."""
        if self._learner is None or not original_text:
            return

        self._finalize_manual_edit_learning()

        now = time.monotonic()
        with self._manual_edit_lock:
            self._manual_edit_session = _ManualEditSession(
                original_text=original_text,
                text=original_text,
                cursor=len(original_text),
                started_at=now,
                last_event_at=now,
            )

    def _discard_manual_edit_monitor(self) -> None:
        """Drop active edit tracking session without learning."""
        with self._manual_edit_lock:
            self._manual_edit_session = None

    def _finalize_manual_edit_learning(self) -> None:
        """Convert inferred manual edits into a learning signal."""
        session: _ManualEditSession | None = None
        with self._manual_edit_lock:
            if self._manual_edit_session is None:
                return
            session = self._manual_edit_session
            self._manual_edit_session = None

        if self._learner is None or session is None or session.edit_count <= 0:
            return

        original = session.original_text
        edited = session.text

        if not edited or original.strip().lower() == edited.strip().lower():
            return

        from rapidfuzz import fuzz

        similarity = fuzz.ratio(original.lower(), edited.lower())
        if similarity < 40:
            logger.info("Skipping manual learn for large text rewrite (%.0f%% similar)", similarity)
            return

        self._last_pasted_text = edited
        self._last_pasted_time = time.monotonic()

        correction_phrase = self._extract_manual_correction_candidate(original, edited)
        if correction_phrase is None:
            return

        logger.info(
            "Manual correction detected (%.0f%% similar): '%s' -> '%s' [candidate: '%s']",
            similarity,
            original,
            edited,
            correction_phrase,
        )
        promoted = self._learner.log_correction(original, correction_phrase)
        if promoted is not None:
            self._reload_corrector_after_learning()
            rumps.notification("Veery", "Learned!", f"Will now correct to: {promoted}")

    def _extract_manual_correction_candidate(self, original: str, edited: str) -> str | None:
        """Extract a short corrected phrase from word-level diffs."""
        original_words = original.split()
        edited_words = edited.split()
        if not original_words or not edited_words:
            return None

        matcher = SequenceMatcher(a=original_words, b=edited_words, autojunk=False)
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal" or j1 == j2:
                continue
            # Include one neighboring word on each side when available.
            start = max(0, j1 - 1)
            end = min(len(edited_words), j2 + 1)
            candidate_words = edited_words[start:end]
            if not candidate_words:
                continue
            if len(candidate_words) > 6:
                continue
            return " ".join(candidate_words)

        if len(edited_words) <= 6:
            return " ".join(edited_words)
        return None

    def _apply_manual_edit_key_locked(self, session: _ManualEditSession, key_kind: str, key_value: str) -> str:
        """Apply a tracked key event to the in-memory edit buffer.

        Returns:
            "tracked": key was interpreted and session updated,
            "ignore": key not relevant,
            "discard": session should be discarded (unknown destructive shortcut).
        """
        cmd_active = any(k in self._modifier_keys_down for k in _CMD_MODIFIERS)
        alt_active = any(k in self._modifier_keys_down for k in _ALT_MODIFIERS)

        if key_kind == "char":
            if cmd_active:
                return "discard"
            if not key_value.isprintable():
                return "ignore"
            self._insert_at_cursor(session, key_value)
            session.edit_count += 1
            return "tracked"

        if key_value == "Key.space":
            if cmd_active:
                return "discard"
            self._insert_at_cursor(session, " ")
            session.edit_count += 1
            return "tracked"

        if key_value in {"Key.enter", "Key.return"}:
            if cmd_active:
                return "discard"
            self._insert_at_cursor(session, "\n")
            session.edit_count += 1
            return "tracked"

        if key_value == "Key.tab":
            if cmd_active:
                return "discard"
            self._insert_at_cursor(session, "\t")
            session.edit_count += 1
            return "tracked"

        if key_value == "Key.left":
            if cmd_active:
                session.cursor = self._line_start(session.text, session.cursor)
            elif alt_active:
                session.cursor = self._prev_word_boundary(session.text, session.cursor)
            else:
                session.cursor = max(0, session.cursor - 1)
            return "tracked"

        if key_value == "Key.right":
            if cmd_active:
                session.cursor = self._line_end(session.text, session.cursor)
            elif alt_active:
                session.cursor = self._next_word_boundary(session.text, session.cursor)
            else:
                session.cursor = min(len(session.text), session.cursor + 1)
            return "tracked"

        if key_value in {"Key.up", "Key.home"}:
            if cmd_active or key_value == "Key.home":
                session.cursor = 0
                return "tracked"
            return "ignore"

        if key_value in {"Key.down", "Key.end"}:
            if cmd_active or key_value == "Key.end":
                session.cursor = len(session.text)
                return "tracked"
            return "ignore"

        if key_value == "Key.backspace":
            if cmd_active:
                start = self._line_start(session.text, session.cursor)
                if start < session.cursor:
                    session.text = session.text[:start] + session.text[session.cursor:]
                    session.cursor = start
                    session.edit_count += 1
                return "tracked"
            if alt_active:
                start = self._prev_word_boundary(session.text, session.cursor)
                if start < session.cursor:
                    session.text = session.text[:start] + session.text[session.cursor:]
                    session.cursor = start
                    session.edit_count += 1
                return "tracked"
            if session.cursor > 0:
                session.text = session.text[: session.cursor - 1] + session.text[session.cursor:]
                session.cursor -= 1
                session.edit_count += 1
            return "tracked"

        if key_value == "Key.delete":
            if cmd_active:
                end = self._line_end(session.text, session.cursor)
                if session.cursor < end:
                    session.text = session.text[:session.cursor] + session.text[end:]
                    session.edit_count += 1
                return "tracked"
            if alt_active:
                end = self._next_word_boundary(session.text, session.cursor)
                if session.cursor < end:
                    session.text = session.text[:session.cursor] + session.text[end:]
                    session.edit_count += 1
                return "tracked"
            if session.cursor < len(session.text):
                session.text = session.text[:session.cursor] + session.text[session.cursor + 1 :]
                session.edit_count += 1
            return "tracked"

        # Cmd + unknown key combo likely means copy/paste/undo etc. We cannot
        # safely reconstruct the text after those edits, so skip this session.
        if cmd_active:
            return "discard"

        return "ignore"

    @staticmethod
    def _insert_at_cursor(session: _ManualEditSession, text: str) -> None:
        session.text = session.text[:session.cursor] + text + session.text[session.cursor:]
        session.cursor += len(text)

    @staticmethod
    def _prev_word_boundary(text: str, cursor: int) -> int:
        i = max(0, min(cursor, len(text)))
        while i > 0 and text[i - 1].isspace():
            i -= 1
        while i > 0 and not text[i - 1].isspace():
            i -= 1
        return i

    @staticmethod
    def _next_word_boundary(text: str, cursor: int) -> int:
        i = max(0, min(cursor, len(text)))
        while i < len(text) and text[i].isspace():
            i += 1
        while i < len(text) and not text[i].isspace():
            i += 1
        return i

    @staticmethod
    def _line_start(text: str, cursor: int) -> int:
        i = max(0, min(cursor, len(text)))
        return text.rfind("\n", 0, i) + 1

    @staticmethod
    def _line_end(text: str, cursor: int) -> int:
        i = max(0, min(cursor, len(text)))
        j = text.find("\n", i)
        return len(text) if j == -1 else j

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: State, *, skip_overlay: bool = False) -> None:
        """Update application state and UI (thread-safe).

        Args:
            new_state: The target state.
            skip_overlay: If True, don't update the overlay (used when
                show_success() was already called and will auto-hide).
        """
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._recording_finalizing = False

        logger.info("State: %s -> %s", old_state.value, new_state.value)

        if new_state == State.IDLE:
            def _ui():
                self.title = "\U0001f3a4"
                self._detail_item.title = "Ready"
                if not skip_overlay:
                    self._overlay.hide()
            _run_on_main_thread(_ui)
            self._cleanup_pending_stt_async()
        elif new_state == State.RECORDING:
            def _ui():
                self.title = "\U0001f534"  # red circle
                self._detail_item.title = "Recording..."
                self._overlay.show_recording()
            _run_on_main_thread(_ui)
        elif new_state == State.PROCESSING:
            def _ui():
                self.title = "\u231b"  # hourglass
                self._detail_item.title = "Processing..."
                self._overlay.show_processing()
            _run_on_main_thread(_ui)

    # ------------------------------------------------------------------
    # Hotkey handler
    # ------------------------------------------------------------------

    def _hotkey_hint(self) -> str:
        """Return the hotkey hint text based on current recording mode."""
        if self._recording_mode == "toggle":
            return f"Press {self._hotkey_label} to start/stop"
        return f"Hold {self._hotkey_label} to dictate"

    def _mode_label(self) -> str:
        """Return the mode menu item label showing the alternative mode."""
        if self._recording_mode == "toggle":
            return "Switch to Hold Mode (hold to dictate)"
        return "Switch to Toggle Mode (press to start/stop)"

    def _on_key_down(self) -> None:
        """Right Cmd pressed — behavior depends on recording mode."""
        if not self._ready.is_set():
            return

        if self._stt is None:
            self._overlay.show_warning("Models loading...")
            return

        # If the user edited the previous output, commit that learning signal
        # before starting the next dictation cycle.
        self._finalize_manual_edit_learning()

        if self._recording_mode == "toggle":
            self._on_toggle_key()
        else:
            self._on_hold_key_down()

    def _on_key_up(self) -> None:
        """Right Cmd released — only used in hold mode."""
        if self._recording_mode == "toggle":
            return  # toggle mode ignores key release

        with self._state_lock:
            if self._state != State.RECORDING:
                return
        self._stop_recording()

    def _begin_recording(self) -> None:
        """Shared logic for starting a recording session (both hold and toggle modes).

        Sets state to RECORDING immediately (before spawning background thread)
        so that key release always sees the correct state. Opens the audio
        stream on the current thread for minimum latency.
        """
        with self._state_lock:
            if self._state != State.IDLE:
                return
            self._state = State.RECORDING  # set immediately to avoid race
            self._recording_finalizing = False
            self._recording_session_id += 1
            session_id = self._recording_session_id

        self._recording_started.clear()
        # Update UI for recording state
        logger.info("State: idle -> recording")

        def _ui():
            self.title = "\U0001f534"  # red circle
            self._detail_item.title = "Recording..."
            self._overlay.show_recording()

        _run_on_main_thread(_ui)
        sounds.play_start()

        if self._recorder is not None:
            try:
                self._recorder.prepare_stream(manual_mode=True)
            except Exception:
                logger.exception("Failed to prepare audio stream")
                self._recording_started.set()  # unblock any waiting _on_key_up
                self._set_state(State.IDLE)
                return
        threading.Thread(target=self._start_recording, args=(session_id,), daemon=True).start()

    def _on_hold_key_down(self) -> None:
        """Hold mode: start recording on key press."""
        self._begin_recording()

    def _on_toggle_key(self) -> None:
        """Toggle mode: press to start recording, press again to stop."""
        with self._state_lock:
            current = self._state
        if current == State.IDLE:
            self._begin_recording()
        elif current == State.RECORDING:
            self._stop_recording()

    def _is_active_recording_session(self, session_id: int | None) -> bool:
        """Return True if a background helper still belongs to the current recording."""
        with self._state_lock:
            return (
                self._state == State.RECORDING
                and not self._recording_finalizing
                and (session_id is None or session_id == self._recording_session_id)
            )

    def _start_recording(self, session_id: int | None = None) -> None:
        """Finalize recording setup (state is already RECORDING, stream may already be open)."""
        if self._recorder is None:
            if self._is_active_recording_session(session_id):
                rumps.notification("Veery", "Error", "Audio recorder not available")
                self._recording_started.set()
                self._set_state(State.IDLE)
            return

        try:
            self._recorder.start_recording(manual_mode=True, open_stream_if_needed=False)
            if not self._is_active_recording_session(session_id):
                return
            self._recording_started.set()
            if self._config.audio.manual_max_duration_sec is not None:
                threading.Thread(target=self._watch_manual_stop, args=(session_id,), daemon=True).start()
        except Exception as e:
            if not self._is_active_recording_session(session_id):
                return
            logger.exception("Failed to start recording")
            self._recording_started.set()  # Unblock key_up even on failure
            rumps.notification("Veery", "Error", str(e))
            self._set_state(State.IDLE)

    def _watch_manual_stop(self, session_id: int | None = None) -> None:
        """Watch for recorder-driven manual stop reasons such as a max-duration cap."""
        recorder = self._recorder
        if recorder is None:
            return

        reason = recorder.wait_for_manual_stop()
        if reason != StopReason.MANUAL_CAP_REACHED:
            return

        if not self._is_active_recording_session(session_id):
            return

        self._stop_recording(reason=reason)

    def _stop_recording(self, *, reason: StopReason = StopReason.USER_STOP) -> None:
        """Stop the current recording and process all captured audio."""
        with self._state_lock:
            if self._state != State.RECORDING or self._recording_finalizing:
                return
            self._recording_finalizing = True

        logger.info("Recording stopped (%s), processing...", reason.value)
        sounds.play_stop()
        if self._recorder is None:
            self._set_state(State.IDLE)
            return

        stop_message = None
        if reason == StopReason.MANUAL_CAP_REACHED:
            stop_message = "Max dictation length reached"
            try:
                rumps.notification("Veery", "Recording stopped", stop_message)
            except Exception:
                logger.warning("Could not send max-duration notification", exc_info=True)

        segment = self._recorder.stop_and_flush(reason=reason)
        if segment is None:
            self._overlay.show_warning(stop_message or "No speech detected")
            self._set_state(State.IDLE, skip_overlay=True)
            return
        # Transition to PROCESSING synchronously so a rapid re-press of the
        # hotkey sees the correct state before the worker thread starts.
        self._set_state(State.PROCESSING)
        worker = threading.Thread(target=self._process_segment, args=(segment,), daemon=True)
        worker.start()

    def _process_segment(self, segment: AudioSegment) -> None:
        """Process a captured audio segment: STT -> correct -> paste.

        Note: state is already PROCESSING (set by _stop_recording before
        spawning this thread) to avoid a race window where rapid hotkey
        presses could re-enter recording.
        """
        success = False
        try:
            if self._recorder is not None:
                try:
                    if not self._recorder.has_speech(segment.audio):
                        logger.info(
                            "Skipping STT: segment failed VAD speech check (duration=%.2fs).",
                            segment.duration_sec,
                        )
                        self._overlay.show_warning("No speech detected")
                        return
                except Exception:
                    logger.exception("Segment VAD speech check failed; continuing to STT")

            # Guard against "press/release with no speech" short noise clips.
            if segment.duration_sec < 1.0:
                rms = float(np.sqrt(np.mean(segment.audio**2)))
                if rms < 0.02:
                    logger.info(
                        "Skipping STT for short low-energy segment (duration=%.2fs, rms=%.4f).",
                        segment.duration_sec,
                        rms,
                    )
                    self._overlay.show_warning("No speech detected")
                    return

            stt = self._stt
            if stt is None:
                rumps.notification("Veery", "Error", "STT model not available")
                return

            raw_text = stt.transcribe(segment.audio, segment.sample_rate)
            if not raw_text:
                self._overlay.show_warning("No speech detected")
                return

            if _is_repetitive_hallucination(raw_text):
                logger.warning("Hallucination detected, discarding: %.80s...", raw_text)
                self._overlay.show_warning("Filtered repetitive audio")
                return

            logger.info("Transcribed: %s", raw_text)

            # Run correction pipeline if available, otherwise use raw text
            if self._corrector is not None:
                result = self._corrector.correct(raw_text)
                final_text = result.final
                logger.info("Final text: %s", final_text)
            else:
                final_text = raw_text

            self._try_auto_learn(final_text)
            paste_to_active_app(final_text, self._config.output)
            self._last_pasted_text = final_text
            self._last_pasted_time = time.monotonic()
            self._begin_manual_edit_monitor(final_text)
            self._session_count += 1
            success = True

        except Exception as e:
            logger.exception("Processing failed")
            rumps.notification("Veery", "Error", str(e))
        finally:
            if success:
                sounds.play_success()
                self._overlay.show_success()
            self._set_state(State.IDLE, skip_overlay=success)
            if self._session_count > 0:
                n = self._session_count
                self._set_detail(
                    f"Ready \u2014 {n} dictation{'s' if n != 1 else ''} this session",
                )

    def _try_auto_learn(self, new_text: str) -> None:
        """If the new dictation is similar to the last one, auto-learn the correction."""
        if self._learner is None or self._last_pasted_text is None:
            return
        elapsed = time.monotonic() - self._last_pasted_time
        if elapsed > self._correction_window_sec:
            return

        # Identical re-dictation is not a correction.
        if self._last_pasted_text.strip().lower() == new_text.strip().lower():
            return

        from rapidfuzz import fuzz

        similarity = fuzz.ratio(self._last_pasted_text.lower(), new_text.lower())
        if similarity < 40:
            # Too different = likely unrelated new dictation
            return

        logger.info(
            "Auto-correction detected (%.0f%% similar, %.1fs ago): '%s' -> '%s'",
            similarity, elapsed, self._last_pasted_text, new_text,
        )
        promoted = self._learner.log_correction(self._last_pasted_text, new_text)
        if promoted is not None:
            self._reload_corrector_after_learning()
            rumps.notification("Veery", "Learned!", f"Will now correct to: {promoted}")

    def _reload_corrector_after_learning(self) -> None:
        """Reload jargon dictionaries so newly promoted terms take effect immediately."""
        try:
            self._corrector = TextCorrector(JargonCorrector(self._config.jargon))
            self._apply_stt_runtime_hints(self._stt)
            logger.info("Reloaded TextCorrector after learning update")
        except Exception:
            logger.exception("Failed to reload TextCorrector after learning update")

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_toggle_mode(self, sender) -> None:
        """Switch between hold-to-talk and press-to-toggle recording modes."""
        self._recording_mode = "hold" if self._recording_mode == "toggle" else "toggle"
        self._hotkey_item.title = self._hotkey_hint()
        self._mode_item.title = self._mode_label()
        logger.info("Recording mode: %s", self._recording_mode)

    def _build_stt_model_submenu(self) -> None:
        """Populate the STT Model submenu with available backends."""
        for backend_id, label in STT_BACKENDS:
            item = rumps.MenuItem(label, callback=functools.partial(self._on_select_stt_backend, backend_id))
            item.state = 1 if backend_id == self._stt_backend else 0
            self._stt_model_menu.add(item)
            self._stt_menu_items[backend_id] = item

    def _on_select_stt_backend(self, backend_id, _sender) -> None:
        """Switch STT backend at runtime."""
        if backend_id == self._stt_backend and self._stt is not None:
            return  # already active

        if self._whisper_loading:
            if backend_id == "whisper":
                # Whisper is already downloading in background — let it auto-switch
                self._whisper_download_cancelled = False
                self._detail_item.title = "Downloading Whisper..."
                logger.info("Whisper already downloading, will auto-switch on completion")
                return
            # Switching away from whisper while it's downloading
            self._whisper_download_cancelled = True

        if self._stt_switching:
            logger.info("STT switch already in progress, ignoring click for %s", backend_id)
            return

        self._stt_switching = True
        self._detail_item.title = "Switching STT model..."
        logger.info("User selected STT backend: %s", backend_id)

        def _switch():
            try:
                from dataclasses import replace
                new_config = replace(self._config.stt, backend=backend_id)
                new_stt = create_stt(new_config)
                self._apply_stt_runtime_hints(new_stt)
                old_stt = self._stt
                self._stt = new_stt
                self._queue_stt_cleanup(old_stt)
                self._stt_backend = backend_id

                def _ui_ok():
                    self._update_stt_checkmarks(backend_id)
                    self._detail_item.title = "Ready"

                _run_on_main_thread(_ui_ok)
                logger.info("STT backend switched to %s", backend_id)
            except Exception:
                logger.exception("Failed to switch STT backend to %s", backend_id)
                self._set_detail("Ready (STT switch failed)")
            finally:
                self._stt_switching = False

        threading.Thread(target=_switch, daemon=True).start()

    def _build_jargon_submenu(self) -> None:
        """Populate the Edit Jargon Dictionary submenu with all active YAML files."""
        from pathlib import Path

        for dict_path_str in self._config.jargon.dict_paths:
            path = Path(dict_path_str)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            item = rumps.MenuItem(
                path.name,
                callback=functools.partial(self._open_jargon_file, path),
            )
            self._edit_jargon_menu.add(item)

        # Include learned.yaml if configured and exists
        if self._config.jargon.learned_path is not None:
            learned = Path(self._config.jargon.learned_path)
            if not learned.is_absolute():
                learned = PROJECT_ROOT / learned
            if learned.exists():
                self._edit_jargon_menu.add(rumps.MenuItem(
                    learned.name,
                    callback=functools.partial(self._open_jargon_file, learned),
                ))

    def _open_jargon_file(self, path, _sender) -> None:
        """Open a jargon YAML file in the default editor."""
        if path.exists():
            subprocess.run(["open", str(path)], check=False)
        else:
            rumps.notification("Veery", "", f"Jargon file not found: {path}")

    def _on_about(self, _sender) -> None:
        """Show About dialog."""
        from AppKit import NSAlert, NSInformationalAlertStyle

        alert = NSAlert.alloc().init()
        alert.setMessageText_(f"Veery v{__version__}")
        alert.setInformativeText_(
            "macOS bilingual dictation for EN/ZH professionals.\n\n"
            "Author: Andy Wang\n"
            "GitHub: github.com/andyhcwang/veery\n"
            "X: x.com/AndyThinkMode"
        )
        alert.setAlertStyle_(NSInformationalAlertStyle)
        alert.addButtonWithTitle_("Close")
        alert.addButtonWithTitle_("Open GitHub")
        alert.addButtonWithTitle_("Open X")

        response = alert.runModal()
        if response == 1001:  # Open GitHub
            webbrowser.open("https://github.com/andyhcwang/veery")
        elif response == 1002:  # Open X
            webbrowser.open("https://x.com/AndyThinkMode")

    def _on_quit(self, _sender) -> None:
        """Clean shutdown."""
        self._finalize_manual_edit_learning()
        self._discard_manual_edit_monitor()
        current_stt = self._stt
        self._stt = None
        self._queue_stt_cleanup(current_stt)
        self._cleanup_pending_stt_resources()
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop_recording()
        rumps.quit_application()


def main() -> None:
    """Entry point for `uv run veery`."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Veery bilingual dictation")
    parser.add_argument("--mine", nargs="+", metavar="PATH",
                        help="Scan paths for potential jargon terms")
    parser.add_argument("--mine-output", metavar="PATH", default="jargon/mined.yaml",
                        help="Output path for mined jargon YAML (default: jargon/mined.yaml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.mine:
        from veery.miner import (
            mine_claude_commands,
            mine_terms,
            print_mining_report,
            write_claude_commands_yaml,
            write_mined_yaml,
        )
        scan_paths = [Path(p).expanduser().resolve() for p in args.mine]
        output_path = Path(args.mine_output).expanduser().resolve()

        # Mine Python terms (existing)
        results = mine_terms(scan_paths)
        written_count = write_mined_yaml(results, output_path, scan_paths)
        print_mining_report(results, written_count=written_count, output_path=output_path)

        # Mine Claude Code custom commands (output next to --mine-output)
        commands = mine_claude_commands(scan_paths)
        if commands:
            cmd_output = output_path.parent / "mined_commands.yaml"
            cmd_written = write_claude_commands_yaml(commands, cmd_output)
            if cmd_written > 0:
                print(f"\nAlso wrote {cmd_written} Claude Code command(s) to {cmd_output}")
        return

    app = VeeryApp()
    app.run()


if __name__ == "__main__":
    main()
