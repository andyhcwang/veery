"""VoiceFlow menubar application. Orchestrates audio capture, STT, and text correction."""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import logging
import subprocess
import threading
import time

import rumps

from voiceflow.audio import AudioRecorder
from voiceflow.config import PROJECT_ROOT, AppConfig, load_config
from voiceflow.corrector import TextCorrector
from voiceflow.grammar import GrammarPolisher
from voiceflow.jargon import JargonCorrector
from voiceflow.learner import CorrectionLearner
from voiceflow.output import paste_to_active_app
from voiceflow.overlay import OverlayIndicator
from voiceflow.stt import SenseVoiceSTT

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


def check_accessibility_permission() -> bool:
    """Check if Accessibility permission is granted (needed for hotkey + text output)."""
    lib_path = ctypes.util.find_library("ApplicationServices")
    if lib_path is None:
        logger.warning("Could not find ApplicationServices library")
        return False
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return bool(lib.AXIsProcessTrusted())


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
        self._status_item = rumps.MenuItem("VoiceFlow \u2014 Ready", callback=None)
        self._status_item.set_callback(None)
        self._edit_jargon_item = rumps.MenuItem("Edit Jargon Dictionary", callback=self._on_edit_jargon)

        self.menu = [
            self._status_item,
            None,  # separator
            self._edit_jargon_item,
            None,  # separator
            rumps.MenuItem("Quit", callback=self._on_quit),
        ]

        # Components (initialized below)
        self._recorder: AudioRecorder | None = None
        self._stt: SenseVoiceSTT | None = None
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

        self._initialize_components()

        # Heavy model loading happens in background after the event loop starts
        threading.Thread(target=self._load_models, daemon=True).start()

    def _initialize_components(self) -> None:
        """Lightweight init — create objects without loading heavy models."""
        logger.info("Initializing VoiceFlow components...")

        # 1. Audio recorder (VAD loaded lazily on first recording)
        try:
            self._recorder = AudioRecorder(self._config.audio, self._config.vad)
            logger.info("AudioRecorder initialized")
        except Exception:
            logger.exception("Failed to initialize AudioRecorder")

        # 2. Jargon (lightweight, no model download)
        try:
            jargon = JargonCorrector(self._config.jargon)
            grammar = GrammarPolisher(self._config.grammar)
            self._corrector = TextCorrector(jargon, grammar)
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

        # 5. Check accessibility permission
        if not check_accessibility_permission():
            logger.warning("Accessibility permission not granted. Opening System Settings...")
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                check=False,
            )

        logger.info("VoiceFlow initialized (loading models in background...)")

    def _load_models(self) -> None:
        """Background thread: download/load heavy models with menubar status."""
        try:
            # STT model
            self.title = "\u2b07"  # down arrow
            self._status_item.title = "Loading STT model..."
            self._stt = SenseVoiceSTT(self._config.stt)
            logger.info("SenseVoiceSTT loaded")

            # Grammar model
            if self._config.grammar.enabled and self._corrector is not None:
                self._status_item.title = "Loading grammar model..."
                self._corrector._grammar.load()
                logger.info("Grammar model loaded")

            self.title = "\U0001f3a4"
            self._status_item.title = "VoiceFlow \u2014 Ready"
            self._ready.set()
            logger.info("All models loaded — ready")
        except Exception:
            logger.exception("Failed to load models")
            self.title = "\u26a0"  # warning
            self._status_item.title = "VoiceFlow \u2014 Model load failed"
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

    def _set_state(self, new_state: State) -> None:
        """Update application state and UI (thread-safe)."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        logger.info("State: %s -> %s", old_state.value, new_state.value)

        if new_state == State.IDLE:
            self.title = "\U0001f3a4"
            self._status_item.title = "VoiceFlow \u2014 Ready"
            self._overlay.hide()
        elif new_state == State.RECORDING:
            self.title = "\U0001f534"  # red circle
            self._status_item.title = "Recording..."
            self._overlay.show_recording()
        elif new_state == State.PROCESSING:
            self.title = "\u231b"  # hourglass
            self._status_item.title = "Processing..."
            self._overlay.show_processing()

    # ------------------------------------------------------------------
    # Hotkey handler
    # ------------------------------------------------------------------

    def _on_key_down(self) -> None:
        """Right Cmd pressed — start recording if idle (non-blocking)."""
        if not self._ready.is_set():
            return
        with self._state_lock:
            if self._state != State.IDLE:
                return
        self._recording_started.clear()
        threading.Thread(target=self._start_recording, daemon=True).start()

    def _on_key_up(self) -> None:
        """Right Cmd released — wait for recording to be ready, then stop and process."""
        with self._state_lock:
            if self._state != State.RECORDING:
                return
        # Wait for recording stream to actually open (handles first-time VAD load)
        self._recording_started.wait(timeout=10)
        self._stop_recording()

    def _start_recording(self) -> None:
        """Begin a new recording session (push-to-talk: stream stays open until key release)."""
        if self._recorder is None:
            rumps.notification("VoiceFlow", "Error", "Audio recorder not available")
            return

        self._set_state(State.RECORDING)
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
                return

            result = self._corrector.correct(raw_text)
            logger.info("Final text: %s", result.final)

            # Auto-learn: if re-dictated within window, treat as correction
            self._try_auto_learn(result.final)

            paste_to_active_app(result.final, self._config.output)
            self._last_pasted_text = result.final
            self._last_pasted_time = time.monotonic()

        except Exception as e:
            logger.exception("Processing failed")
            rumps.notification("VoiceFlow", "Error", str(e))
        finally:
            self._set_state(State.IDLE)

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

    def _on_edit_jargon(self, _sender) -> None:
        """Open the quant_finance.yaml jargon file in the default editor."""
        jargon_path = PROJECT_ROOT / "jargon" / "quant_finance.yaml"
        if jargon_path.exists():
            subprocess.run(["open", str(jargon_path)], check=False)
        else:
            rumps.notification("VoiceFlow", "", f"Jargon file not found: {jargon_path}")

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
