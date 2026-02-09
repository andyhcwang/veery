"""Speech-to-text backends: SenseVoice-Small (FunASR) and Whisper (mlx-whisper)."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from veery.config import STTConfig

logger = logging.getLogger(__name__)

_STALL_TIMEOUT_SEC = 45
_MAX_DOWNLOAD_SEC = 1200  # 20 min safety net


class DownloadStalled(Exception):
    """Download progress stopped for too long."""

# Regex to strip SenseVoice special tokens: language, emotion, and event tags.
# Examples: <|zh|>, <|en|>, <|HAPPY|>, <|BGM|>, <|Speech|>, <|Applause|>
_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")


def _format_download_detail(name: str, downloaded_bytes: float, total_bytes: float) -> str:
    """Format download progress as a human-readable string (MB or GB)."""
    dl_mb = downloaded_bytes / (1024 * 1024)
    tot_mb = total_bytes / (1024 * 1024)
    if tot_mb >= 1024:
        return f"Downloading {name} ({dl_mb / 1024:.1f} / {tot_mb / 1024:.1f} GB)..."
    return f"Downloading {name} ({dl_mb:.0f} / {tot_mb:.0f} MB)..."


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


def _clean_incomplete_cache(repo_id: str) -> int:
    """Delete ``.incomplete`` blobs and ``.lock`` files from HF cache for a model.

    Returns the count of deleted files.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return 0

    model_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if not model_dir.is_dir():
        return 0

    deleted = 0
    blobs = model_dir / "blobs"
    if blobs.is_dir():
        for f in blobs.iterdir():
            if f.suffix in (".incomplete", ".lock"):
                try:
                    f.unlink()
                    deleted += 1
                    logger.debug("Deleted incomplete blob: %s", f)
                except OSError:
                    logger.debug("Could not delete %s", f)
    logger.info("Cleaned %d incomplete/lock files for %s", deleted, repo_id)
    return deleted


def _snapshot_download_with_stall_detection(
    repo_id: str,
    progress_callback: Callable[[float, str], None] | None = None,
    *,
    env_override: dict[str, str] | None = None,
) -> None:
    """Run ``snapshot_download`` in a thread with stall detection.

    Raises:
        DownloadStalled: If no download progress is made for ``_STALL_TIMEOUT_SEC``.
        TimeoutError: If total time exceeds ``_MAX_DOWNLOAD_SEC``.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import _tqdm as hf_tqdm_module
    from huggingface_hub.utils import tqdm as OriginalTqdm

    # Shared state between download thread and monitor.
    # bytes_downloaded is cumulative across all files to avoid false stall
    # detection when a new tqdm instance resets per-file progress to 0.
    # cancelled is set by the monitor to signal the download thread to abort.
    state = {"bytes_downloaded": 0, "error": None, "done": False, "cancelled": False}
    state_lock = threading.Lock()

    class _ProgressTqdm(OriginalTqdm):
        """Wraps HF's tqdm to forward progress and track bytes for stall detection."""

        def update(self, n=1):
            super().update(n)
            with state_lock:
                if state["cancelled"]:
                    raise DownloadStalled("Download cancelled by stall detector")
                state["bytes_downloaded"] += n
            if self.total and self.total > 0 and progress_callback is not None:
                fraction = self.n / self.total
                detail = _format_download_detail(repo_id.split("/")[-1], self.n, self.total)
                progress_callback(fraction, detail)

    def _download():
        old_env = {}
        if env_override:
            for k, v in env_override.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v
        hf_tqdm_module.tqdm = _ProgressTqdm
        try:
            snapshot_download(repo_id)
        except Exception as exc:
            with state_lock:
                state["error"] = exc
        finally:
            hf_tqdm_module.tqdm = OriginalTqdm
            if env_override:
                for k in env_override:
                    if old_env.get(k) is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old_env[k]
            with state_lock:
                state["done"] = True

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()

    # Monitor loop: check for stalls every 2 seconds
    last_bytes = 0
    last_progress_time = time.monotonic()
    start_time = last_progress_time

    while True:
        thread.join(timeout=2.0)

        with state_lock:
            done = state["done"]
            error = state["error"]
            current_bytes = state["bytes_downloaded"]

        if done:
            if error is not None:
                raise error
            return

        now = time.monotonic()

        if current_bytes > last_bytes:
            last_bytes = current_bytes
            last_progress_time = now

        if now - last_progress_time > _STALL_TIMEOUT_SEC:
            logger.warning("Download stalled for %s (no progress for %ds)", repo_id, _STALL_TIMEOUT_SEC)
            with state_lock:
                state["cancelled"] = True
            raise DownloadStalled(f"No progress for {_STALL_TIMEOUT_SEC}s")

        if now - start_time > _MAX_DOWNLOAD_SEC:
            logger.warning("Download timed out for %s (exceeded %ds)", repo_id, _MAX_DOWNLOAD_SEC)
            with state_lock:
                state["cancelled"] = True
            raise TimeoutError(f"Download exceeded {_MAX_DOWNLOAD_SEC}s")


def _curl_download_hf_model(
    repo_id: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    """Download a HF model's weights via curl, then finalize with snapshot_download.

    Uses ``HfApi().list_repo_tree()`` to find the largest file (weights),
    downloads it with curl, places it in the HF blob cache by SHA256, then
    runs ``snapshot_download`` with xet disabled to create metadata/symlinks.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.constants import HF_HUB_CACHE

    api = HfApi()

    # Find the largest file in the repo (the weights file)
    largest_file = None
    largest_size = 0
    for entry in api.list_repo_tree(repo_id, recursive=True):
        if hasattr(entry, "size") and entry.size is not None and entry.size > largest_size:
            largest_size = entry.size
            largest_file = entry.rfilename

    if largest_file is None:
        raise RuntimeError(f"Could not find any files in {repo_id}")

    logger.info("Downloading %s/%s (%d bytes) via curl", repo_id, largest_file, largest_size)

    # Build download URL
    url = f"https://huggingface.co/{repo_id}/resolve/main/{largest_file}"

    model_name = repo_id.split("/")[-1]
    if progress_callback:
        progress_callback(0.0, f"Downloading {model_name} via curl...")

    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["curl", "-L", "-f", "--retry", "3", "-o", tmp_path, url],
            capture_output=True,
            text=True,
            timeout=_MAX_DOWNLOAD_SEC,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr[:200]}")

        # Verify file size
        actual_size = Path(tmp_path).stat().st_size
        if actual_size < largest_size * 0.9:
            raise RuntimeError(f"Downloaded file too small: {actual_size} vs expected {largest_size}")

        if progress_callback:
            progress_callback(0.8, f"Verifying {model_name}...")

        # Compute SHA256 and place in blob cache
        sha256 = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
        blob_hash = sha256.hexdigest()

        model_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)

        blob_path = blobs_dir / blob_hash
        if not blob_path.exists():
            shutil.move(tmp_path, blob_path)
            logger.info("Placed weights blob at %s", blob_path)
        else:
            Path(tmp_path).unlink()
            logger.info("Blob %s already exists, skipping", blob_hash)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    if progress_callback:
        progress_callback(0.9, f"Finalizing {model_name}...")

    # Run snapshot_download with xet disabled to create metadata/symlinks
    old_xet = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id)
    finally:
        if old_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = old_xet


def ensure_model_downloaded(
    repo_id: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    """Pre-download a HuggingFace model with three-tier retry and stall detection.

    1. Normal ``snapshot_download`` with stall timeout
    2. Retry with ``HF_HUB_DISABLE_XET=1`` (disables problematic xet CDN)
    3. ``curl`` fallback for weights, then ``snapshot_download`` for metadata

    Args:
        repo_id: HuggingFace model repo (e.g., "mlx-community/whisper-large-v3-turbo").
        progress_callback: Called with (fraction 0.0-1.0, detail_string).
    """
    if _is_model_cached(repo_id):
        logger.info("Model %s already cached, skipping download", repo_id)
        return

    logger.info("Downloading model %s ...", repo_id)

    if progress_callback is not None:
        progress_callback(0.0, f"Downloading {repo_id}...")

    # Attempt 1: normal snapshot_download with stall detection
    try:
        _snapshot_download_with_stall_detection(repo_id, progress_callback)
        if progress_callback is not None:
            progress_callback(1.0, "Download complete")
        logger.info("Model %s downloaded (attempt 1)", repo_id)
        return
    except Exception as exc:
        logger.warning("Download attempt 1 failed for %s: %s", repo_id, exc)
        _clean_incomplete_cache(repo_id)
        if progress_callback is not None:
            progress_callback(0.0, "Download stalled, retrying...")

    # Attempt 2: retry with xet CDN disabled
    try:
        _snapshot_download_with_stall_detection(
            repo_id,
            progress_callback,
            env_override={"HF_HUB_DISABLE_XET": "1"},
        )
        if progress_callback is not None:
            progress_callback(1.0, "Download complete")
        logger.info("Model %s downloaded (attempt 2, xet disabled)", repo_id)
        return
    except Exception as exc:
        logger.warning("Download attempt 2 failed for %s: %s", repo_id, exc)
        _clean_incomplete_cache(repo_id)
        if progress_callback is not None:
            progress_callback(0.0, "Retrying with direct download...")

    # Attempt 3: curl fallback
    _curl_download_hf_model(repo_id, progress_callback)
    if progress_callback is not None:
        progress_callback(1.0, "Download complete")
    logger.info("Model %s downloaded (attempt 3, curl fallback)", repo_id)


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
                detail = _format_download_detail(
                    model_name.split("/")[-1], state["downloaded_bytes"], state["total_bytes"]
                )
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
