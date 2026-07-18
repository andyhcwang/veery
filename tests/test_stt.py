"""Tests for WhisperSTT backend and create_stt factory."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from veery.config import STTConfig


@pytest.fixture
def cached_whisper_model():
    """Prevent Whisper unit tests from consulting or downloading a real model."""
    with patch("veery.stt._is_model_cached", return_value=True):
        yield

# ---------------------------------------------------------------------------
# WhisperSTT
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cached_whisper_model")
class TestWhisperSTT:
    def test_transcribe_success(self) -> None:
        """Mock mlx_whisper.transcribe, verify WhisperSTT returns cleaned text."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": "  hello world  "}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == "hello world"
        mock_sf.write.assert_called_once()
        mock_mlx.transcribe.assert_called_once()
        call_kwargs = mock_mlx.transcribe.call_args.kwargs
        assert call_kwargs["condition_on_previous_text"] is False
        assert call_kwargs["no_speech_threshold"] == 0.6
        assert call_kwargs["logprob_threshold"] == -1.0
        assert call_kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        assert call_kwargs["hallucination_silence_threshold"] == 0.2

    def test_transcribe_empty_audio(self) -> None:
        """Empty audio array returns ''."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        result = stt.transcribe(np.array([], dtype=np.float32), 16000)
        assert result == ""

    def test_transcribe_exception_returns_empty(self) -> None:
        """When mlx_whisper.transcribe raises, WhisperSTT returns ''."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.side_effect = RuntimeError("model crashed")

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == ""

    def test_transcribe_uses_fresh_temp_files_and_unlinks_them(self) -> None:
        """Each call gets an isolated WAV path that is removed afterward."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        import soundfile as sf

        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": "hello"}

        # Patch sys.modules so the lazy `import mlx_whisper` / `import soundfile`
        # inside transcribe() resolves to our mocks/real modules.
        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": sf}):
            result1 = stt.transcribe(audio, 16000)
            result2 = stt.transcribe(audio, 16000)

        assert result1 == "hello"
        assert result2 == "hello"
        calls = mock_mlx.transcribe.call_args_list
        paths = [call.args[0] for call in calls]
        assert paths[0] != paths[1]
        assert stt._tmp_wav not in paths
        assert all(not Path(path).exists() for path in paths)

    def test_transcribe_temp_file_is_unlinked_after_error(self) -> None:
        """The per-call WAV is removed even when transcription raises."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        created_paths: list[str] = []
        mock_mlx = MagicMock()

        def capture_and_raise(path, **kwargs):
            created_paths.append(path)
            raise RuntimeError("fail")

        mock_mlx.transcribe.side_effect = capture_and_raise

        import soundfile as sf

        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": sf}):
            result = stt.transcribe(audio, 16000)

        assert result == ""
        assert len(created_paths) == 1
        assert created_paths[0] != stt._tmp_wav
        assert Path(created_paths[0]).exists() is False

    def test_transcribe_returns_early_when_model_not_cached(self) -> None:
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.ones(16000, dtype=np.float32)
        mock_sf = MagicMock()
        mock_mlx = MagicMock()

        with (
            patch("veery.stt._is_model_cached", return_value=False),
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == ""
        mock_sf.write.assert_not_called()
        mock_mlx.transcribe.assert_not_called()

    def test_load_model_warmup(self) -> None:
        """load_model() transcribes a silent wav to warm up, sets _loaded."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        assert not stt._loaded

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            stt.load_model()

        assert stt._loaded
        mock_mlx.transcribe.assert_called_once()
        # Verify model name is passed
        call_kwargs = mock_mlx.transcribe.call_args
        assert call_kwargs[1]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"

    def test_load_model_already_loaded_skips(self) -> None:
        """Second load_model() call is a no-op."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        stt._loaded = True

        mock_mlx = MagicMock()
        with patch("veery.stt.mlx_whisper", mock_mlx, create=True):
            stt.load_model()

        mock_mlx.transcribe.assert_not_called()

    def test_load_model_exception_keeps_unloaded(self) -> None:
        """If warmup fails, _loaded stays False."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.side_effect = RuntimeError("download failed")

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            stt.load_model()

        assert not stt._loaded

    def test_transcribe_result_not_dict(self) -> None:
        """If mlx_whisper returns a non-dict, fallback to str()."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = "raw string result"

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == "raw string result"

    def test_transcribe_skips_low_energy_clip(self) -> None:
        """Short low-energy clips should be treated as no-speech."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.zeros(16000, dtype=np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == ""
        mock_sf.write.assert_not_called()
        mock_mlx.transcribe.assert_not_called()

    def test_transcribe_zero_sample_rate_returns_empty(self) -> None:
        """Zero sample rate should safely return '' without raising."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.ones(16000, dtype=np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 0)

        assert result == ""
        mock_sf.write.assert_not_called()
        mock_mlx.transcribe.assert_not_called()

    def test_default_config(self) -> None:
        """WhisperSTT with no config uses STTConfig defaults."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT()
        assert stt._config.backend == "whisper"
        assert stt._config.whisper_model == "mlx-community/whisper-large-v3-turbo"

    def test_energy_gate_uses_relaxed_thresholds(self) -> None:
        from veery.stt import WhisperSTT

        assert WhisperSTT._MIN_RMS_ENERGY == 0.003
        assert WhisperSTT._MIN_ACTIVE_FRAME_RMS == 0.008

    def test_transcribe_passes_language_and_prompt_hints(self) -> None:
        """Configured language and jargon prompt should be forwarded to mlx-whisper."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(
            STTConfig(
                backend="whisper",
                language="en",
                whisper_initial_prompt="Static prompt.",
            )
        )
        stt.set_runtime_hints(prompt="Runtime jargon.")
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": "hello world"}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == "hello world"
        call_kwargs = mock_mlx.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "en"
        assert call_kwargs["initial_prompt"] == "Static prompt. Runtime jargon."

    def test_release_resources_clears_holder_and_caches(self) -> None:
        """release_resources() should clear the cached MLX model and flush caches."""
        from veery.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper", whisper_model="mlx-community/whisper-small"))
        holder = SimpleNamespace(model=object(), model_path="mlx-community/whisper-small")
        transcribe_module = SimpleNamespace(ModelHolder=holder)

        with (
            patch("veery.stt.importlib.import_module", return_value=transcribe_module),
            patch.dict(sys.modules, {"mlx": MagicMock()}),
        ):
            stt.release_resources()

        assert holder.model is None
        assert holder.model_path is None
        assert stt._loaded is False


# ---------------------------------------------------------------------------
# create_stt factory
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cached_whisper_model")
class TestCreateSTT:
    def test_create_stt_sensevoice(self) -> None:
        """backend='sensevoice' returns SenseVoiceSTT."""
        from veery.stt import SenseVoiceSTT, create_stt

        with patch("veery.stt.SenseVoiceSTT._load_model"):
            stt = create_stt(STTConfig(backend="sensevoice"))

        assert isinstance(stt, SenseVoiceSTT)

    def test_create_stt_whisper(self) -> None:
        """backend='whisper' returns WhisperSTT with load_model called."""
        from veery.stt import WhisperSTT, create_stt

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("veery.stt.mlx_whisper", mock_mlx, create=True),
            patch("veery.stt.sf", mock_sf, create=True),
        ):
            stt = create_stt(STTConfig(backend="whisper"))

        assert isinstance(stt, WhisperSTT)
        assert stt._loaded

    def test_create_stt_default_is_whisper(self) -> None:
        """Default config creates WhisperSTT."""
        from veery.stt import WhisperSTT, create_stt

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}
        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}):
            stt = create_stt()

        assert isinstance(stt, WhisperSTT)
        assert stt._loaded is True

    def test_create_stt_invalid_backend_falls_back_to_whisper(self, caplog) -> None:
        """Unknown backends warn and use Whisper."""
        from veery.stt import WhisperSTT, create_stt

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}
        with (
            caplog.at_level(logging.WARNING),
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
        ):
            stt = create_stt(STTConfig(backend="nonexistent"))

        assert isinstance(stt, WhisperSTT)
        assert stt._config.backend == "whisper"
        assert "using whisper" in caplog.text

    def test_create_stt_whisper_raises_when_model_not_cached(self) -> None:
        from veery.stt import create_stt

        with (
            patch("veery.stt._is_model_cached", return_value=False),
            pytest.raises(RuntimeError, match="Whisper backend failed to load"),
        ):
            create_stt(STTConfig(backend="whisper"))

    def test_sensevoice_load_failure_is_raised(self) -> None:
        from veery.stt import SenseVoiceSTT

        mock_funasr = MagicMock()
        mock_funasr.AutoModel.side_effect = RuntimeError("model load failed")
        with (
            patch.dict("sys.modules", {"funasr": mock_funasr}),
            pytest.raises(RuntimeError, match="model load failed"),
        ):
            SenseVoiceSTT(STTConfig(backend="sensevoice"))


# ---------------------------------------------------------------------------
# STTConfig fields
# ---------------------------------------------------------------------------


class TestSTTConfigDefaults:
    def test_stt_config_defaults(self) -> None:
        """Verify default values for STTConfig fields."""
        cfg = STTConfig()
        assert cfg.backend == "whisper"
        assert cfg.whisper_model == "mlx-community/whisper-large-v3-turbo"
        assert cfg.model_name == "iic/SenseVoiceSmall"
        assert cfg.language == "auto"
        assert cfg.device == "cpu"
        assert cfg.whisper_use_jargon_prompt is True
        assert cfg.whisper_prompt_terms_limit == 64
        assert cfg.whisper_prompt_char_limit == 400
        assert cfg.whisper_initial_prompt is None
        assert cfg.processing_timeout_sec == 120.0

    def test_stt_config_custom_values(self) -> None:
        """STTConfig accepts custom overrides."""
        cfg = STTConfig(
            backend="whisper",
            whisper_model="mlx-community/whisper-small",
            whisper_initial_prompt="Company glossary.",
            whisper_prompt_terms_limit=12,
        )
        assert cfg.backend == "whisper"
        assert cfg.whisper_model == "mlx-community/whisper-small"
        assert cfg.whisper_initial_prompt == "Company glossary."
        assert cfg.whisper_prompt_terms_limit == 12


# ---------------------------------------------------------------------------
# _is_model_cached
# ---------------------------------------------------------------------------


class TestWhisperCacheCheck:
    def test_incomplete_blob_invalidates_otherwise_complete_cache(self, tmp_path: Path) -> None:
        from veery.stt import _is_model_cached

        model_dir = tmp_path / "models--org--name"
        snapshot = model_dir / "snapshots" / "revision"
        blobs = model_dir / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        (snapshot / "config.json").write_text("{}")
        (blobs / "bighash").write_bytes(b"x" * 1_000_001)
        (blobs / "foo.incomplete").touch()

        with patch("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path)):
            assert _is_model_cached("org/name") is False


# ---------------------------------------------------------------------------
# _is_sensevoice_cached
# ---------------------------------------------------------------------------


class TestSenseVoiceCacheCheck:
    def test_is_sensevoice_cached_exists(self, tmp_path) -> None:
        """Returns True when model.pt exists at expected cache path."""
        from veery.stt import _is_sensevoice_cached

        model_dir = tmp_path / ".cache" / "modelscope" / "hub" / "models" / "iic" / "SenseVoiceSmall"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").touch()

        with patch("veery.stt.Path.home", return_value=tmp_path):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is True

    def test_is_sensevoice_cached_not_exists(self, tmp_path) -> None:
        """Returns False when model.pt does not exist."""
        from veery.stt import _is_sensevoice_cached

        with patch("veery.stt.Path.home", return_value=tmp_path):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is False

    def test_is_sensevoice_cached_exception(self) -> None:
        """Returns False when Path.home() raises an exception."""
        from veery.stt import _is_sensevoice_cached

        with patch("veery.stt.Path.home", side_effect=RuntimeError("no home")):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is False


# ---------------------------------------------------------------------------
# ensure_sensevoice_downloaded
# ---------------------------------------------------------------------------


class TestEnsureSenseVoiceDownloaded:
    def test_already_cached_skips_download(self) -> None:
        """When model is cached, snapshot_download is NOT called."""
        from veery.stt import ensure_sensevoice_downloaded

        mock_snapshot = MagicMock()
        with (
            patch("veery.stt._is_sensevoice_cached", return_value=True),
            patch.dict("sys.modules", {"modelscope.hub.snapshot_download": MagicMock(snapshot_download=mock_snapshot)}),
        ):
            ensure_sensevoice_downloaded("iic/SenseVoiceSmall")

        mock_snapshot.assert_not_called()

    def test_not_cached_calls_snapshot_download(self) -> None:
        """When model is NOT cached, snapshot_download is called with model name."""
        from veery.stt import ensure_sensevoice_downloaded

        mock_snapshot = MagicMock()
        mock_ms_module = MagicMock()
        mock_ms_module.snapshot_download = mock_snapshot

        with (
            patch("veery.stt._is_sensevoice_cached", return_value=False),
            patch.dict("sys.modules", {
                "modelscope": MagicMock(),
                "modelscope.hub": MagicMock(),
                "modelscope.hub.snapshot_download": mock_ms_module,
            }),
        ):
            callback = MagicMock()
            ensure_sensevoice_downloaded("iic/SenseVoiceSmall", progress_callback=callback)

        mock_snapshot.assert_called_once()
        call_args = mock_snapshot.call_args
        assert call_args[0][0] == "iic/SenseVoiceSmall"
        assert "progress_callbacks" in call_args[1]
        assert len(call_args[1]["progress_callbacks"]) == 1
        # progress_callback should have been called for start (0.0) and end (1.0)
        callback.assert_any_call(0.0, "Downloading SenseVoiceSmall...")
        callback.assert_any_call(1.0, "Download complete")


# ---------------------------------------------------------------------------
# _clean_incomplete_cache
# ---------------------------------------------------------------------------


class TestCleanIncompleteCache:
    def test_removes_incomplete_and_lock_files(self, tmp_path) -> None:
        """Deletes .incomplete blobs and .lock files from the model blob dir."""
        from veery.stt import _clean_incomplete_cache

        model_dir = tmp_path / "models--org--model" / "blobs"
        model_dir.mkdir(parents=True)
        (model_dir / "abc123.incomplete").touch()
        (model_dir / "def456.lock").touch()
        (model_dir / "good_blob").touch()

        mock_constants = MagicMock()
        mock_constants.HF_HUB_CACHE = str(tmp_path)
        with patch.dict("sys.modules", {"huggingface_hub.constants": mock_constants}):
            deleted = _clean_incomplete_cache("org/model")

        assert deleted == 2
        assert not (model_dir / "abc123.incomplete").exists()
        assert not (model_dir / "def456.lock").exists()
        assert (model_dir / "good_blob").exists()

    def test_noop_when_model_dir_missing(self, tmp_path) -> None:
        """Returns 0 when the model cache directory doesn't exist."""
        from veery.stt import _clean_incomplete_cache

        mock_constants = MagicMock()
        mock_constants.HF_HUB_CACHE = str(tmp_path)
        with patch.dict("sys.modules", {"huggingface_hub.constants": mock_constants}):
            deleted = _clean_incomplete_cache("org/nonexistent")

        assert deleted == 0


# ---------------------------------------------------------------------------
# _snapshot_download_with_stall_detection
# ---------------------------------------------------------------------------


class TestStallDetection:
    def test_raises_on_stall(self) -> None:
        """Raises DownloadStalled when progress stops for too long."""
        from veery.stt import DownloadStalled, _snapshot_download_with_stall_detection

        def fake_snapshot_download(repo_id):
            # Simulate a stall: sleep longer than stall timeout
            time.sleep(10)

        mock_tqdm_module = MagicMock()
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = fake_snapshot_download
        mock_hf_hub_utils = MagicMock()
        mock_hf_hub_utils._tqdm = mock_tqdm_module
        mock_hf_hub_utils.tqdm = MagicMock

        with (
            patch("veery.stt._STALL_TIMEOUT_SEC", 1),
            patch.dict("sys.modules", {
                "huggingface_hub": mock_hf_hub,
                "huggingface_hub.utils": mock_hf_hub_utils,
                "huggingface_hub.utils._tqdm": mock_tqdm_module,
            }),
        ):
            with pytest.raises(DownloadStalled):
                _snapshot_download_with_stall_detection("org/model")

    def test_completes_normally(self) -> None:
        """Normal download completes without triggering stall detection."""
        from veery.stt import _snapshot_download_with_stall_detection

        def fast_download(repo_id):
            pass  # completes instantly

        mock_tqdm_module = MagicMock()
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = fast_download
        mock_hf_hub_utils = MagicMock()
        mock_hf_hub_utils._tqdm = mock_tqdm_module
        mock_hf_hub_utils.tqdm = MagicMock

        with patch.dict("sys.modules", {
            "huggingface_hub": mock_hf_hub,
            "huggingface_hub.utils": mock_hf_hub_utils,
            "huggingface_hub.utils._tqdm": mock_tqdm_module,
        }):
            # Should not raise
            _snapshot_download_with_stall_detection("org/model")


# ---------------------------------------------------------------------------
# ensure_model_downloaded (three-tier retry)
# ---------------------------------------------------------------------------


class TestEnsureModelDownloaded:
    def test_cached_model_skips_download(self) -> None:
        """When model is already cached, no download occurs."""
        from veery.stt import ensure_model_downloaded

        callback = MagicMock()
        with patch("veery.stt._is_model_cached", return_value=True):
            ensure_model_downloaded("org/model", progress_callback=callback)

        callback.assert_not_called()

    def test_first_attempt_succeeds(self) -> None:
        """When first attempt succeeds, no retries happen."""
        from veery.stt import ensure_model_downloaded

        callback = MagicMock()
        with (
            patch("veery.stt._is_model_cached", return_value=False),
            patch("veery.stt._snapshot_download_with_stall_detection") as mock_sd,
            patch("veery.stt._clean_incomplete_cache") as mock_clean,
        ):
            ensure_model_downloaded("org/model", progress_callback=callback)

        mock_sd.assert_called_once_with("org/model", callback)
        mock_clean.assert_not_called()
        callback.assert_any_call(1.0, "Download complete")

    def test_stall_triggers_xet_disabled_retry(self) -> None:
        """When first attempt stalls, retries with HF_HUB_DISABLE_XET=1."""
        from veery.stt import DownloadStalled, ensure_model_downloaded

        callback = MagicMock()

        call_count = {"n": 0}
        def fake_stall_detect(repo_id, cb, *, env_override=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise DownloadStalled("stalled")
            # Second call succeeds

        with (
            patch("veery.stt._is_model_cached", return_value=False),
            patch("veery.stt._snapshot_download_with_stall_detection", side_effect=fake_stall_detect) as mock_sd,
            patch("veery.stt._clean_incomplete_cache") as mock_clean,
        ):
            ensure_model_downloaded("org/model", progress_callback=callback)

        assert mock_sd.call_count == 2
        # Second call should have env_override for xet disable
        second_call = mock_sd.call_args_list[1]
        assert second_call.kwargs["env_override"] == {"HF_HUB_DISABLE_XET": "1"}
        mock_clean.assert_called_once()
        callback.assert_any_call(0.0, "Download stalled, retrying...")
        callback.assert_any_call(1.0, "Download complete")

    def test_both_stalls_trigger_curl_fallback(self) -> None:
        """When both snapshot_download attempts stall, falls back to curl."""
        from veery.stt import DownloadStalled, ensure_model_downloaded

        callback = MagicMock()

        with (
            patch("veery.stt._is_model_cached", return_value=False),
            patch("veery.stt._snapshot_download_with_stall_detection", side_effect=DownloadStalled("stalled")),
            patch("veery.stt._clean_incomplete_cache") as mock_clean,
            patch("veery.stt._curl_download_hf_model") as mock_curl,
        ):
            ensure_model_downloaded("org/model", progress_callback=callback)

        assert mock_clean.call_count == 2
        mock_curl.assert_called_once_with("org/model", callback)
        callback.assert_any_call(0.0, "Retrying with direct download...")
        callback.assert_any_call(1.0, "Download complete")
