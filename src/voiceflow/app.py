"""VoiceFlow menubar application. Orchestrates audio capture, STT, and text correction."""

from __future__ import annotations

import enum
import functools
import logging
import subprocess
import threading
import time
import webbrowser

import rumps

from voiceflow import __version__
from voiceflow.audio import AudioRecorder
from voiceflow.config import PROJECT_ROOT, STT_BACKENDS, AppConfig, load_config
from voiceflow.corrector import TextCorrector
from voiceflow.jargon import JargonCorrector
from voiceflow.learner import CorrectionLearner
from voiceflow.output import paste_to_active_app
from voiceflow.overlay import (
    DownloadProgressOverlay,
    OverlayIndicator,
    PermissionGuideOverlay,
    check_permissions_granted,
)
from voiceflow.stt import (
    SenseVoiceSTT,
    WhisperSTT,
    _is_model_cached,
    create_stt,
    ensure_model_downloaded,
)

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class VoiceFlowApp(rumps.App):
    """macOS menubar app for bilingual dictation with jargon correction."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()

        super().__init__(
            name="VoiceFlow",
            title="\U0001f3a4",  # microphone emoji
            quit_button=None,  # We add our own quit to control ordering
        )

        # State machine
        self._state = State.IDLE
        self._state_lock = threading.Lock()

        # Menu items
        self._status_item = rumps.MenuItem(f"VoiceFlow v{__version__}", callback=None)
        self._status_item.set_callback(None)

        # Status detail (shows model state or last action)
        self._detail_item = rumps.MenuItem("Loading models...", callback=None)
        self._detail_item.set_callback(None)

        # Recording mode: "hold" (push-to-talk) or "toggle" (press-to-toggle)
        self._recording_mode: str = self._config.hotkey.mode

        # Hotkey info (updates based on mode)
        self._hotkey_item = rumps.MenuItem(self._hotkey_hint(), callback=None)
        self._hotkey_item.set_callback(None)

        # Recording mode toggle
        self._mode_item = rumps.MenuItem(
            "Toggle Mode (press to start/stop)",
            callback=self._on_toggle_mode,
        )
        self._mode_item.state = 1 if self._recording_mode == "toggle" else 0

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
            rumps.MenuItem("About VoiceFlow", callback=self._on_about),
            None,  # separator
            rumps.MenuItem("Quit VoiceFlow", callback=self._on_quit),
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

        # Last pasted text + timestamp for auto-correction detection
        self._last_pasted_text: str | None = None
        self._last_pasted_time: float = 0.0
        self._correction_window_sec: float = 30.0  # auto-learn if re-dictated within 30s

        # Signals that recording stream is actually open and capturing audio
        self._recording_started = threading.Event()

        # Signals that all models are loaded and app is ready
        self._ready = threading.Event()

        self._permission_overlay = PermissionGuideOverlay()

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Lightweight init — create objects without loading heavy models."""
        logger.info("Initializing VoiceFlow components...")

        # 1. Audio recorder (VAD loaded lazily on first recording)
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

        logger.info("VoiceFlow initialized")

    def _on_permissions_granted(self) -> None:
        """Called when all permissions are granted (from PermissionGuideOverlay)."""
        logger.info("Permissions granted, starting model loading...")
        self._detail_item.title = "Loading models..."
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self) -> None:
        """Background thread: download/load heavy models with menubar status.

        Shows a DownloadProgressOverlay when models need to be downloaded for the
        first time. If models are already cached, skips the overlay entirely.
        """
        download_overlay: DownloadProgressOverlay | None = None
        try:
            # VAD model (small, ~50MB — load first so recording is instant)
            self.title = "\u2b07"  # down arrow
            if self._recorder is not None:
                self._detail_item.title = "Loading VAD model..."
                self._recorder._ensure_vad_loaded()
                logger.info("VAD model pre-loaded")

            # Check if STT model needs downloading
            stt_cfg = self._config.stt
            repo_id = stt_cfg.model_name if stt_cfg.backend != "whisper" else stt_cfg.whisper_model
            needs_download = not _is_model_cached(repo_id)

            if needs_download:
                download_overlay = DownloadProgressOverlay()
                download_overlay.show()
                self._detail_item.title = "Downloading model..."

                def _progress(fraction: float, detail: str) -> None:
                    if download_overlay is not None:
                        download_overlay.set_progress(fraction, detail)

                ensure_model_downloaded(repo_id, progress_callback=_progress)

                # Update overlay to show warmup phase
                download_overlay.set_progress(1.0, "Loading model into memory...")
            else:
                self._detail_item.title = "Loading STT model..."

            # STT model
            self._stt = create_stt(stt_cfg)
            logger.info("STT loaded (backend=%s)", stt_cfg.backend)

            # Hide download overlay if it was shown
            if download_overlay is not None:
                download_overlay.hide()

            # Ready for recording
            self.title = "\U0001f3a4"
            self._detail_item.title = "Ready"
            self._ready.set()
            logger.info("STT ready — accepting dictation")
        except Exception:
            logger.exception("Failed to load models")
            if download_overlay is not None:
                download_overlay.hide()
            self.title = "\u26a0"  # warning
            self._detail_item.title = "Model load failed"
            self._ready.set()  # unblock anyway

    def _start_hotkey_listener(self) -> None:
        """Start push-to-talk listener: hold right Cmd to record, release to process."""
        try:
            from pynput.keyboard import Key, Listener

            logger.info("Registering push-to-talk key: right Cmd")

            def on_press(key):
                if key == Key.cmd_r:
                    self._on_key_down()

            def on_release(key):
                if key == Key.cmd_r:
                    self._on_key_up()

            self._hotkey_listener = Listener(on_press=on_press, on_release=on_release)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception:
            logger.exception("Failed to start hotkey listener")

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

        logger.info("State: %s -> %s", old_state.value, new_state.value)

        if new_state == State.IDLE:
            self.title = "\U0001f3a4"
            self._detail_item.title = "Ready"
            if not skip_overlay:
                self._overlay.hide()
        elif new_state == State.RECORDING:
            self.title = "\U0001f534"  # red circle
            self._detail_item.title = "Recording..."
            self._overlay.show_recording()
        elif new_state == State.PROCESSING:
            self.title = "\u231b"  # hourglass
            self._detail_item.title = "Processing..."
            self._overlay.show_processing()

    # ------------------------------------------------------------------
    # Hotkey handler
    # ------------------------------------------------------------------

    def _hotkey_hint(self) -> str:
        """Return the hotkey hint text based on current recording mode."""
        if self._recording_mode == "toggle":
            return "Press Right \u2318 to start/stop"
        return "Hold Right \u2318 to dictate"

    def _on_key_down(self) -> None:
        """Right Cmd pressed — behavior depends on recording mode."""
        if not self._ready.is_set():
            return

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
        self._recording_started.wait(timeout=10)
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

        self._recording_started.clear()
        # Update UI for recording state
        logger.info("State: idle -> recording")
        self.title = "\U0001f534"  # red circle
        self._detail_item.title = "Recording..."
        self._overlay.show_recording()

        if self._recorder is not None:
            try:
                self._recorder.prepare_stream()
            except Exception:
                logger.exception("Failed to prepare audio stream")
                self._set_state(State.IDLE)
                return
        threading.Thread(target=self._start_recording, daemon=True).start()

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
            self._recording_started.wait(timeout=10)
            self._stop_recording()

    def _start_recording(self) -> None:
        """Finalize recording setup (state is already RECORDING, stream may already be open)."""
        if self._recorder is None:
            rumps.notification("VoiceFlow", "Error", "Audio recorder not available")
            self._recording_started.set()
            self._set_state(State.IDLE)
            return

        try:
            self._recorder.start_recording()
            self._recording_started.set()
        except Exception as e:
            logger.exception("Failed to start recording")
            self._recording_started.set()  # Unblock key_up even on failure
            rumps.notification("VoiceFlow", "Error", str(e))
            self._set_state(State.IDLE)

    def _stop_recording(self) -> None:
        """Stop the current recording and process all captured audio."""
        logger.info("Recording stopped by user, processing...")
        if self._recorder is None:
            self._set_state(State.IDLE)
            return
        segment = self._recorder.stop_and_flush()
        if segment is None:
            rumps.notification("VoiceFlow", "", "No speech detected")
            self._set_state(State.IDLE)
            return
        # Process on a background thread to avoid blocking the hotkey listener
        worker = threading.Thread(target=self._process_segment, args=(segment,), daemon=True)
        worker.start()

    def _process_segment(self, segment) -> None:
        """Process a captured audio segment: STT -> correct -> paste."""
        self._set_state(State.PROCESSING)
        success = False
        try:
            if self._stt is None:
                rumps.notification("VoiceFlow", "Error", "STT model not available")
                return

            raw_text = self._stt.transcribe(segment.audio, segment.sample_rate)
            if not raw_text:
                rumps.notification("VoiceFlow", "", "Could not transcribe audio")
                return

            logger.info("Transcribed: %s", raw_text)

            if self._corrector is None:
                self._try_auto_learn(raw_text)
                paste_to_active_app(raw_text, self._config.output)
                self._last_pasted_text = raw_text
                self._last_pasted_time = time.monotonic()
                self._session_count += 1
                success = True
                return

            result = self._corrector.correct(raw_text)
            logger.info("Final text: %s", result.final)

            # Auto-learn: if re-dictated within window, treat as correction
            self._try_auto_learn(result.final)

            paste_to_active_app(result.final, self._config.output)
            self._last_pasted_text = result.final
            self._last_pasted_time = time.monotonic()
            self._session_count += 1
            success = True

        except Exception as e:
            logger.exception("Processing failed")
            rumps.notification("VoiceFlow", "Error", str(e))
        finally:
            if success:
                # Show success flash — it auto-hides after 800ms
                self._overlay.show_success()
            self._set_state(State.IDLE, skip_overlay=success)
            if self._session_count > 0:
                n = self._session_count
                self._detail_item.title = f"Ready \u2014 {n} dictation{'s' if n != 1 else ''} this session"

    def _try_auto_learn(self, new_text: str) -> None:
        """If the new dictation is similar to the last one, auto-learn the correction."""
        if self._learner is None or self._last_pasted_text is None:
            return
        elapsed = time.monotonic() - self._last_pasted_time
        if elapsed > self._correction_window_sec:
            return

        from rapidfuzz import fuzz

        similarity = fuzz.ratio(self._last_pasted_text.lower(), new_text.lower())
        if similarity < 40 or similarity > 95:
            # Too different = new text, too similar = same thing repeated
            return

        logger.info(
            "Auto-correction detected (%.0f%% similar, %.1fs ago): '%s' -> '%s'",
            similarity, elapsed, self._last_pasted_text, new_text,
        )
        promoted = self._learner.log_correction(self._last_pasted_text, new_text)
        if promoted is not None:
            rumps.notification("VoiceFlow", "Learned!", f"Will now correct to: {promoted}")

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_toggle_mode(self, sender) -> None:
        """Switch between hold-to-talk and press-to-toggle recording modes."""
        sender.state = not sender.state
        self._recording_mode = "toggle" if sender.state else "hold"
        self._hotkey_item.title = self._hotkey_hint()
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

        # Update checkmarks
        for bid, item in self._stt_menu_items.items():
            item.state = 1 if bid == backend_id else 0

        self._detail_item.title = "Switching STT model..."
        logger.info("User selected STT backend: %s", backend_id)

        def _switch():
            try:
                from dataclasses import replace
                new_config = replace(self._config.stt, backend=backend_id)
                new_stt = create_stt(new_config)
                self._stt = new_stt
                self._stt_backend = backend_id
                self._detail_item.title = "Ready"
                logger.info("STT backend switched to %s", backend_id)
            except Exception:
                logger.exception("Failed to switch STT backend to %s", backend_id)
                self._detail_item.title = "Ready (STT switch failed)"

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
            rumps.notification("VoiceFlow", "", f"Jargon file not found: {path}")

    def _on_about(self, _sender) -> None:
        """Show About dialog."""
        response = rumps.alert(
            title=f"VoiceFlow v{__version__}",
            message=(
                "macOS bilingual dictation for EN/ZH professionals.\n\n"
                "Author: Cedric Wang\n"
                "GitHub: github.com/shahdpinky/voiceflow\n"
                "Email: haocheng.c.wang@gmail.com"
            ),
            ok="Close",
            other="Open GitHub",
        )
        if response == 0:  # "Other" button
            webbrowser.open("https://github.com/shahdpinky/voiceflow")

    def _on_quit(self, _sender) -> None:
        """Clean shutdown."""
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop_recording()
        rumps.quit_application()


def main() -> None:
    """Entry point for `uv run voiceflow`."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="VoiceFlow bilingual dictation")
    parser.add_argument("--mine", nargs="+", metavar="PATH",
                        help="Scan paths for potential jargon terms")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.mine:
        from voiceflow.miner import mine_terms, print_mining_report
        scan_paths = [Path(p).expanduser().resolve() for p in args.mine]
        results = mine_terms(scan_paths)
        print_mining_report(results)
        return

    app = VoiceFlowApp()
    app.run()


if __name__ == "__main__":
    main()
