"""Speech-to-text backends: SenseVoice-Small (FunASR) and Whisper (mlx-whisper)."""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from veery.config import STTConfig

logger = logging.getLogger(__name__)

# Regex to strip SenseVoice special tokens: language, emotion, and event tags.
# Examples: <|zh|>, <|en|>, <|HAPPY|>, <|BGM|>, <|Speech|>, <|Applause|>
_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")


def _is_sensevoice_cached(model_name: str) -> bool:
    """Check if a ModelScope model (e.g. SenseVoice) is already cached locally.

    ModelScope stores models at ~/.cache/modelscope/hub/models/<org>/<name>/model.pt.
    """
    try:
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models"
        # model_name is slash-separated like "iic/SenseVoiceSmall"
        model_dir = cache_dir.joinpath(*model_name.split("/"))
        return (model_dir / "model.pt").exists()
    except Exception:
        logger.debug("Could not check ModelScope cache for %s", model_name)
        return False


def _is_model_cached(repo_id: str) -> bool:
    """Check if a HuggingFace model is already fully cached locally.

    Uses a direct path check instead of scan_cache_dir() to avoid
    walking the entire HF cache directory (50-200ms on large caches).
    Verifies that at least one large blob (>1MB) exists without an
    ``.incomplete`` suffix, to avoid false positives from partial downloads.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        model_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        if not model_dir.is_dir():
            return False
        # A valid cached model has at least a snapshots/ dir with content
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            return False
        # Verify blobs contain at least one large complete file (weights).
        # Incomplete downloads have a .incomplete suffix.
        blobs = model_dir / "blobs"
        if not blobs.is_dir():
            return False
        for blob in blobs.iterdir():
            if blob.suffix == ".incomplete":
                continue
            if blob.is_file() and blob.stat().st_size > 1_000_000:
                return True
        return False
    except Exception:
        logger.debug("Could not check HF cache for %s", repo_id)
    return False


def ensure_model_downloaded(
    repo_id: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    """Pre-download a HuggingFace model with progress tracking.

    If the model is already cached, this returns immediately. Otherwise it
    downloads via ``huggingface_hub.snapshot_download()`` and reports progress
    through the callback.

    Args:
        repo_id: HuggingFace model repo (e.g., "iic/SenseVoiceSmall").
        progress_callback: Called with (fraction 0.0-1.0, detail_string).
    """
    if _is_model_cached(repo_id):
        logger.info("Model %s already cached, skipping download", repo_id)
        return

    logger.info("Downloading model %s ...", repo_id)

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import _tqdm as hf_tqdm_module
    from huggingface_hub.utils import tqdm as OriginalTqdm

    if progress_callback is not None:
        progress_callback(0.0, f"Downloading {repo_id}...")

    # Monkey-patch tqdm at the internal module level to capture download progress.
    # huggingface_hub.utils._tqdm is the module; its .tqdm attribute is the class.
    class _ProgressTqdm(OriginalTqdm):
        """Wraps HF's tqdm to forward progress to our callback."""

        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0 and progress_callback is not None:
                fraction = self.n / self.total
                downloaded_mb = self.n / (1024 * 1024)
                total_mb = self.total / (1024 * 1024)
                if total_mb >= 1024:
                    name = repo_id.split("/")[-1]
                    dl_gb = downloaded_mb / 1024
                    tot_gb = total_mb / 1024
                    detail = f"Downloading {name} ({dl_gb:.1f} / {tot_gb:.1f} GB)..."
                else:
                    detail = f"Downloading {repo_id.split('/')[-1]} ({downloaded_mb:.0f} / {total_mb:.0f} MB)..."
                progress_callback(fraction, detail)

    hf_tqdm_module.tqdm = _ProgressTqdm
    try:
        snapshot_download(repo_id)
    finally:
        hf_tqdm_module.tqdm = OriginalTqdm

    if progress_callback is not None:
        progress_callback(1.0, "Download complete")
    logger.info("Model %s downloaded", repo_id)


def ensure_sensevoice_downloaded(
    model_name: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    """Pre-download a ModelScope model (SenseVoice) with progress tracking.

    If the model is already cached, returns immediately. Otherwise downloads
    via ``modelscope.hub.snapshot_download()`` and reports aggregate progress.

    Args:
        model_name: ModelScope model ID (e.g., "iic/SenseVoiceSmall").
        progress_callback: Called with (fraction 0.0-1.0, detail_string).
    """
    if _is_sensevoice_cached(model_name):
        logger.info("SenseVoice model %s already cached, skipping download", model_name)
        return

    logger.info("Downloading SenseVoice model %s ...", model_name)

    if progress_callback is not None:
        progress_callback(0.0, f"Downloading {model_name.split('/')[-1]}...")

    from modelscope.hub.snapshot_download import snapshot_download

    # Track aggregate progress across all files via closure
    state = {"total_bytes": 0, "downloaded_bytes": 0}

    class _AggregateProgress:
        """ModelScope ProgressCallback subclass that aggregates per-file progress."""

        def __init__(self, filename: str, file_size: int) -> None:
            self._filename = filename
            self._file_size = file_size
            state["total_bytes"] += file_size

        def update(self, size: int) -> None:
            state["downloaded_bytes"] += size
            if progress_callback is not None and state["total_bytes"] > 0:
                fraction = min(state["downloaded_bytes"] / state["total_bytes"], 1.0)
                name = model_name.split("/")[-1]
                dl_mb = state["downloaded_bytes"] / (1024 * 1024)
                tot_mb = state["total_bytes"] / (1024 * 1024)
                if tot_mb >= 1024:
                    detail = f"Downloading {name} ({dl_mb / 1024:.1f} / {tot_mb / 1024:.1f} GB)..."
                else:
                    detail = f"Downloading {name} ({dl_mb:.0f} / {tot_mb:.0f} MB)..."
                progress_callback(fraction, detail)

        def end(self) -> None:
            pass

    snapshot_download(model_name, progress_callbacks=[_AggregateProgress])

    if progress_callback is not None:
        progress_callback(1.0, "Download complete")
    logger.info("SenseVoice model %s downloaded", model_name)


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
    """Wrapper around mlx-whisper for multilingual-robust STT on Apple Silicon."""

    def __init__(self, config: STTConfig | None = None) -> None:
        self._config = config or STTConfig()
        self._loaded = False
        # Pre-allocate a single temp WAV file reused across transcriptions
        # to avoid create/delete overhead on every call (~5-15ms savings).
        _f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_wav: str = _f.name
        _f.close()
        import atexit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Remove the temporary WAV file."""
        try:
            Path(self._tmp_wav).unlink(missing_ok=True)
        except Exception:
            pass

    def load_model(self) -> None:
        """Warm up the Whisper model by running a dummy transcription.

        mlx-whisper loads the model lazily on first transcribe() call.
        We trigger that here so the first real transcription is fast.
        """
        if self._loaded:
            return
        try:
            import mlx_whisper
            import soundfile as sf

            logger.info("Loading Whisper model: %s", self._config.whisper_model)
            # Create a short silent WAV to trigger model download/load
            sf.write(self._tmp_wav, np.zeros(16000, dtype=np.float32), 16000)
            mlx_whisper.transcribe(self._tmp_wav, path_or_hf_repo=self._config.whisper_model)
            self._loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception:
            logger.exception("Failed to load Whisper model")

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

        try:
            import mlx_whisper
            import soundfile as sf

            # Reuse pre-allocated temp WAV path (mlx-whisper expects a file path)
            sf.write(self._tmp_wav, audio, sample_rate)

            result = mlx_whisper.transcribe(self._tmp_wav, path_or_hf_repo=self._config.whisper_model)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return text.strip()
        except Exception:
            logger.exception("Whisper transcription failed")
            return ""


def create_stt(config: STTConfig | None = None) -> SenseVoiceSTT | WhisperSTT:
    """Factory: return the right STT backend based on config.backend."""
    config = config or STTConfig()
    if config.backend == "whisper":
        stt = WhisperSTT(config)
        stt.load_model()
        return stt
    return SenseVoiceSTT(config)
