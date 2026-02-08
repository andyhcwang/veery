"""VoiceFlow menubar application. Orchestrates audio capture, STT, and text correction."""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import logging
import subprocess
import threading

import rumps

from voiceflow.audio import AudioRecorder
from voiceflow.config import PROJECT_ROOT, AppConfig, load_config
from voiceflow.corrector import TextCorrector
from voiceflow.grammar import GrammarPolisher
from voiceflow.jargon import JargonCorrector
from voiceflow.learner import CorrectionLearner
from voiceflow.output import paste_to_active_app
from voiceflow.stt import SenseVoiceSTT

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


# Map config key names to pynput modifier format
_MODIFIER_MAP: dict[str, str] = {
    "cmd": "<cmd>",
    "ctrl": "<ctrl>",
    "alt": "<alt>",
    "shift": "<shift>",
}


def _parse_hotkey(combo: str) -> str:
    """Convert config hotkey string (e.g. 'cmd+shift+space') to pynput format ('<cmd>+<shift>+<space>')."""
    parts = combo.lower().split("+")
    converted = []
    for part in parts:
        part = part.strip()
        if part in _MODIFIER_MAP:
            converted.append(_MODIFIER_MAP[part])
        else:
            # Regular key — wrap in angle brackets for pynput
            converted.append(f"<{part}>")
    return "+".join(converted)


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

        # Last pasted text for correction mode
        self._last_pasted_text: str | None = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Load all pipeline components. Heavy models are loaded eagerly (VAD + STT)."""
        logger.info("Initializing VoiceFlow components...")

        # 1. Audio recorder (loads VAD eagerly)
        try:
            self._recorder = AudioRecorder(self._config.audio, self._config.vad)
            logger.info("AudioRecorder initialized")
        except Exception:
            logger.exception("Failed to initialize AudioRecorder")

        # 2. STT (loads model eagerly)
        try:
            self._stt = SenseVoiceSTT(self._config.stt)
            logger.info("SenseVoiceSTT initialized")
        except Exception:
            logger.exception("Failed to initialize SenseVoiceSTT")

        # 3. Jargon + Grammar -> TextCorrector
        try:
            jargon = JargonCorrector(self._config.jargon)
            grammar = GrammarPolisher(self._config.grammar)
            self._corrector = TextCorrector(jargon, grammar)
            logger.info("TextCorrector initialized")
        except Exception:
            logger.exception("Failed to initialize TextCorrector")

        # 4. Correction learner
        if self._config.learning.enabled:
            try:
                self._learner = CorrectionLearner(self._config.learning)
                logger.info("CorrectionLearner initialized")
            except Exception:
                logger.exception("Failed to initialize CorrectionLearner")

        # 5. Global hotkey listener
        self._start_hotkey_listener()

        # 6. Check accessibility permission
        if not check_accessibility_permission():
            logger.warning("Accessibility permission not granted. Opening System Settings...")
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                check=False,
            )

        logger.info("VoiceFlow initialization complete")

    def _start_hotkey_listener(self) -> None:
        """Start the global hotkey listener on a daemon thread (pynput manages threading)."""
        try:
            from pynput.keyboard import GlobalHotKeys

            hotkey_str = _parse_hotkey(self._config.hotkey.key_combo)
            logger.info("Registering global hotkey: %s -> %s", self._config.hotkey.key_combo, hotkey_str)

            hotkeys = {hotkey_str: self._on_hotkey}

            # Register correction hotkey if learning is enabled
            if self._config.learning.enabled:
                correction_str = _parse_hotkey(self._config.learning.correction_hotkey)
                logger.info(
                    "Registering correction hotkey: %s -> %s",
                    self._config.learning.correction_hotkey,
                    correction_str,
                )
                hotkeys[correction_str] = self._on_correction_hotkey

            self._hotkey_listener = GlobalHotKeys(hotkeys)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception:
            logger.exception("Failed to start global hotkey listener")

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
        elif new_state == State.RECORDING:
            self.title = "\U0001f534"  # red circle
            self._status_item.title = "Recording..."
        elif new_state == State.PROCESSING:
            self.title = "\u231b"  # hourglass
            self._status_item.title = "Processing..."

    # ------------------------------------------------------------------
    # Hotkey handler
    # ------------------------------------------------------------------

    def _on_hotkey(self) -> None:
        """Handle global hotkey press. Called from pynput thread."""
        with self._state_lock:
            current = self._state

        if current == State.IDLE:
            self._start_recording()
        elif current == State.RECORDING:
            self._cancel_recording()
        # PROCESSING: ignore hotkey (v1 simplicity)

    def _on_correction_hotkey(self) -> None:
        """Handle correction hotkey press. Records a short correction and learns from it."""
        if self._last_pasted_text is None:
            logger.info("Correction hotkey pressed but no previous text to correct")
            return

        with self._state_lock:
            if self._state != State.IDLE:
                logger.info("Correction hotkey ignored, not in IDLE state")
                return

        if self._recorder is None or self._stt is None or self._learner is None:
            rumps.notification("VoiceFlow", "Error", "Components not available for correction")
            return

        self._set_state(State.RECORDING)
        try:
            self._recorder.start_recording()
            worker = threading.Thread(target=self._correction_worker, daemon=True)
            worker.start()
        except Exception as e:
            logger.exception("Failed to start correction recording")
            rumps.notification("VoiceFlow", "Error", str(e))
            self._set_state(State.IDLE)

    def _correction_worker(self) -> None:
        """Worker thread: record correction phrase, transcribe, and learn."""
        if self._recorder is None or self._stt is None or self._learner is None:
            self._set_state(State.IDLE)
            return

        try:
            segment = self._recorder.wait_for_speech_end(timeout=self._config.audio.max_duration_sec)
        except Exception:
            logger.exception("Error waiting for correction speech end")
            self._set_state(State.IDLE)
            return

        with self._state_lock:
            if self._state != State.RECORDING:
                return

        if segment is None:
            rumps.notification("VoiceFlow", "", "No correction speech detected")
            self._set_state(State.IDLE)
            return

        self._set_state(State.PROCESSING)
        try:
            correction_text = self._stt.transcribe(segment.audio, segment.sample_rate)
            if not correction_text:
                rumps.notification("VoiceFlow", "", "Could not transcribe correction")
                return

            logger.info("Correction transcribed: %s", correction_text)

            promoted = self._learner.log_correction(self._last_pasted_text, correction_text)
            if promoted is not None:
                rumps.notification("VoiceFlow", "Learned!", f"Will now correct to: {promoted}")
        except Exception:
            logger.exception("Correction processing failed")
        finally:
            self._set_state(State.IDLE)

    def _start_recording(self) -> None:
        """Begin a new recording session."""
        if self._recorder is None:
            rumps.notification("VoiceFlow", "Error", "Audio recorder not available")
            return

        self._set_state(State.RECORDING)
        try:
            self._recorder.start_recording()
            # Spawn a daemon worker thread to wait for speech end
            worker = threading.Thread(target=self._recording_worker, daemon=True)
            worker.start()
        except Exception as e:
            logger.exception("Failed to start recording")
            rumps.notification("VoiceFlow", "Error", str(e))
            self._set_state(State.IDLE)

    def _cancel_recording(self) -> None:
        """Cancel the current recording (hotkey pressed during recording)."""
        logger.info("Recording cancelled by user")
        self._set_state(State.IDLE)
        if self._recorder is not None:
            self._recorder.stop_recording()  # Stops stream and signals done event

    def _recording_worker(self) -> None:
        """Worker thread: wait for VAD to detect end-of-speech, then process."""
        if self._recorder is None:
            return

        try:
            segment = self._recorder.wait_for_speech_end(timeout=self._config.audio.max_duration_sec)
        except Exception:
            logger.exception("Error waiting for speech end")
            self._set_state(State.IDLE)
            return

        # Check if we're still in RECORDING state (might have been cancelled)
        with self._state_lock:
            if self._state != State.RECORDING:
                return

        if segment is None:
            # No speech detected or too short -- timeout with no speech
            rumps.notification("VoiceFlow", "", "No speech detected")
            self._set_state(State.IDLE)
            return

        # Speech detected, move to processing
        self._process_segment(segment)

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
                # Fallback: paste raw text without correction
                paste_to_active_app(raw_text, self._config.output)
                self._last_pasted_text = raw_text
                return

            result = self._corrector.correct(raw_text)
            logger.info("Final text: %s", result.final)
            paste_to_active_app(result.final, self._config.output)
            self._last_pasted_text = result.final

        except Exception as e:
            logger.exception("Processing failed")
            rumps.notification("VoiceFlow", "Error", str(e))
        finally:
            self._set_state(State.IDLE)

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
