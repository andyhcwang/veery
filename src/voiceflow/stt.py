"""Speech-to-text backends: SenseVoice-Small (FunASR) and Whisper (lightning-whisper-mlx)."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np

from voiceflow.config import STTConfig

logger = logging.getLogger(__name__)

# Regex to strip SenseVoice special tokens: language, emotion, and event tags.
# Examples: <|zh|>, <|en|>, <|HAPPY|>, <|BGM|>, <|Speech|>, <|Applause|>
_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")


class SenseVoiceSTT:
    """Wrapper around FunASR's SenseVoice-Small model for bilingual STT."""

    def __init__(self, config: STTConfig | None = None) -> None:
        self._config = config or STTConfig()
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Eagerly load the SenseVoice model at init time."""
        try:
            from funasr import AutoModel

            logger.info(
                "Loading SenseVoice model: %s (device=%s)",
                self._config.model_name,
                self._config.device,
            )
            self._model = AutoModel(
                model=self._config.model_name,
                device=self._config.device,
            )
            logger.info("SenseVoice model loaded successfully")
        except Exception:
            logger.exception("Failed to load SenseVoice model")
            self._model = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array to text.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz (SenseVoice expects 16kHz).

        Returns:
            Cleaned transcription string, or "" on error.
        """
        if self._model is None:
            logger.error("SenseVoice model not loaded, returning empty string")
            return ""

        if audio.size == 0:
            return ""

        try:
            result = self._model.generate(
                input=audio,
                language=self._config.language,
                use_itn=True,
            )
            return _extract_text(result)
        except Exception:
            logger.exception("SenseVoice transcription failed")
            return ""


def _extract_text(result: list) -> str:
    """Extract and clean text from FunASR generate() output.

    FunASR returns a list of dicts, each with a 'text' key. SenseVoice
    prepends language/emotion/event tags like <|zh|><|NEUTRAL|><|Speech|>.
    """
    if not result:
        return ""

    raw_text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
    return _strip_tags(raw_text)


def _strip_tags(text: str) -> str:
    """Remove all SenseVoice special tokens and clean whitespace."""
    cleaned = _TAG_PATTERN.sub("", text)
    return cleaned.strip()


class WhisperSTT:
    """Wrapper around mlx-whisper for accent-robust multilingual STT on Apple Silicon."""

    def __init__(self, config: STTConfig | None = None) -> None:
        self._config = config or STTConfig()
        self._loaded = False

    def load_model(self) -> None:
        """Warm up the Whisper model by running a dummy transcription.

        mlx-whisper loads the model lazily on first transcribe() call.
        We trigger that here so the first real transcription is fast.
        """
        if self._loaded:
            return
        tmp_path: str | None = None
        try:
            import mlx_whisper
            import soundfile as sf

            logger.info("Loading Whisper model: %s", self._config.whisper_model)
            # Create a short silent WAV to trigger model download/load
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, np.zeros(16000, dtype=np.float32), 16000)
            mlx_whisper.transcribe(tmp_path, path_or_hf_repo=self._config.whisper_model)
            self._loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception:
            logger.exception("Failed to load Whisper model")
        finally:
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array to text.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Transcription string, or "" on error.
        """
        if audio.size == 0:
            return ""

        tmp_path: str | None = None
        try:
            import mlx_whisper
            import soundfile as sf

            # Write audio to a temporary WAV file (mlx-whisper expects a file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, sample_rate)

            result = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=self._config.whisper_model)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return text.strip()
        except Exception:
            logger.exception("Whisper transcription failed")
            return ""
        finally:
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass


def create_stt(config: STTConfig | None = None) -> SenseVoiceSTT | WhisperSTT:
    """Factory: return the right STT backend based on config.backend."""
    config = config or STTConfig()
    if config.backend == "whisper":
        stt = WhisperSTT(config)
        stt.load_model()
        return stt
    return SenseVoiceSTT(config)
