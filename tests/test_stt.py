"""Tests for WhisperSTT backend and create_stt factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from voiceflow.config import STTConfig

# ---------------------------------------------------------------------------
# WhisperSTT
# ---------------------------------------------------------------------------


class TestWhisperSTT:
    def test_transcribe_success(self) -> None:
        """Mock mlx_whisper.transcribe, verify WhisperSTT returns cleaned text."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": "  hello world  "}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == "hello world"
        mock_sf.write.assert_called_once()
        mock_mlx.transcribe.assert_called_once()

    def test_transcribe_empty_audio(self) -> None:
        """Empty audio array returns ''."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        result = stt.transcribe(np.array([], dtype=np.float32), 16000)
        assert result == ""

    def test_transcribe_exception_returns_empty(self) -> None:
        """When mlx_whisper.transcribe raises, WhisperSTT returns ''."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.side_effect = RuntimeError("model crashed")

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == ""

    def test_transcribe_temp_file_reused(self, tmp_path) -> None:
        """Verify the same pre-allocated temp wav path is reused across calls."""
        from voiceflow.stt import WhisperSTT

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
        # Both calls should use the same pre-allocated temp path
        calls = mock_mlx.transcribe.call_args_list
        assert calls[0][0][0] == calls[1][0][0]
        assert calls[0][0][0] == stt._tmp_wav

    def test_transcribe_temp_file_survives_error(self) -> None:
        """Temp file persists for reuse even when transcription raises."""
        from voiceflow.stt import WhisperSTT

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
        # The pre-allocated temp path should be the same as the instance's
        assert created_paths[0] == stt._tmp_wav

    def test_load_model_warmup(self) -> None:
        """load_model() transcribes a silent wav to warm up, sets _loaded."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        assert not stt._loaded

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            stt.load_model()

        assert stt._loaded
        mock_mlx.transcribe.assert_called_once()
        # Verify model name is passed
        call_kwargs = mock_mlx.transcribe.call_args
        assert call_kwargs[1]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"

    def test_load_model_already_loaded_skips(self) -> None:
        """Second load_model() call is a no-op."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        stt._loaded = True

        mock_mlx = MagicMock()
        with patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True):
            stt.load_model()

        mock_mlx.transcribe.assert_not_called()

    def test_load_model_exception_keeps_unloaded(self) -> None:
        """If warmup fails, _loaded stays False."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.side_effect = RuntimeError("download failed")

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            stt.load_model()

        assert not stt._loaded

    def test_transcribe_result_not_dict(self) -> None:
        """If mlx_whisper returns a non-dict, fallback to str()."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT(STTConfig(backend="whisper"))
        audio = np.random.randn(16000).astype(np.float32)

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = "raw string result"

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            result = stt.transcribe(audio, 16000)

        assert result == "raw string result"

    def test_default_config(self) -> None:
        """WhisperSTT with no config uses STTConfig defaults."""
        from voiceflow.stt import WhisperSTT

        stt = WhisperSTT()
        assert stt._config.backend == "whisper"
        assert stt._config.whisper_model == "mlx-community/whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# create_stt factory
# ---------------------------------------------------------------------------


class TestCreateSTT:
    def test_create_stt_sensevoice(self) -> None:
        """backend='sensevoice' returns SenseVoiceSTT."""
        from voiceflow.stt import SenseVoiceSTT, create_stt

        with patch("voiceflow.stt.SenseVoiceSTT._load_model"):
            stt = create_stt(STTConfig(backend="sensevoice"))

        assert isinstance(stt, SenseVoiceSTT)

    def test_create_stt_whisper(self) -> None:
        """backend='whisper' returns WhisperSTT with load_model called."""
        from voiceflow.stt import WhisperSTT, create_stt

        mock_sf = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": ""}

        with (
            patch.dict("sys.modules", {"mlx_whisper": mock_mlx, "soundfile": mock_sf}),
            patch("voiceflow.stt.mlx_whisper", mock_mlx, create=True),
            patch("voiceflow.stt.sf", mock_sf, create=True),
        ):
            stt = create_stt(STTConfig(backend="whisper"))

        assert isinstance(stt, WhisperSTT)
        assert stt._loaded

    def test_create_stt_default_is_whisper(self) -> None:
        """Default config creates WhisperSTT."""
        from voiceflow.stt import WhisperSTT, create_stt

        with patch("voiceflow.stt.WhisperSTT.load_model"):
            stt = create_stt()

        assert isinstance(stt, WhisperSTT)

    def test_create_stt_invalid_backend_returns_sensevoice(self) -> None:
        """Unknown backend falls through to SenseVoiceSTT (current behavior)."""
        from voiceflow.stt import SenseVoiceSTT, create_stt

        with patch("voiceflow.stt.SenseVoiceSTT._load_model"):
            stt = create_stt(STTConfig(backend="nonexistent"))

        assert isinstance(stt, SenseVoiceSTT)


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

    def test_stt_config_custom_values(self) -> None:
        """STTConfig accepts custom overrides."""
        cfg = STTConfig(
            backend="whisper",
            whisper_model="mlx-community/whisper-small",
        )
        assert cfg.backend == "whisper"
        assert cfg.whisper_model == "mlx-community/whisper-small"


# ---------------------------------------------------------------------------
# _is_sensevoice_cached
# ---------------------------------------------------------------------------


class TestSenseVoiceCacheCheck:
    def test_is_sensevoice_cached_exists(self, tmp_path) -> None:
        """Returns True when model.pt exists at expected cache path."""
        from voiceflow.stt import _is_sensevoice_cached

        model_dir = tmp_path / ".cache" / "modelscope" / "hub" / "models" / "iic" / "SenseVoiceSmall"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").touch()

        with patch("voiceflow.stt.Path.home", return_value=tmp_path):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is True

    def test_is_sensevoice_cached_not_exists(self, tmp_path) -> None:
        """Returns False when model.pt does not exist."""
        from voiceflow.stt import _is_sensevoice_cached

        with patch("voiceflow.stt.Path.home", return_value=tmp_path):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is False

    def test_is_sensevoice_cached_exception(self) -> None:
        """Returns False when Path.home() raises an exception."""
        from voiceflow.stt import _is_sensevoice_cached

        with patch("voiceflow.stt.Path.home", side_effect=RuntimeError("no home")):
            assert _is_sensevoice_cached("iic/SenseVoiceSmall") is False


# ---------------------------------------------------------------------------
# ensure_sensevoice_downloaded
# ---------------------------------------------------------------------------


class TestEnsureSenseVoiceDownloaded:
    def test_already_cached_skips_download(self) -> None:
        """When model is cached, snapshot_download is NOT called."""
        from voiceflow.stt import ensure_sensevoice_downloaded

        mock_snapshot = MagicMock()
        with (
            patch("voiceflow.stt._is_sensevoice_cached", return_value=True),
            patch.dict("sys.modules", {"modelscope.hub.snapshot_download": MagicMock(snapshot_download=mock_snapshot)}),
        ):
            ensure_sensevoice_downloaded("iic/SenseVoiceSmall")

        mock_snapshot.assert_not_called()

    def test_not_cached_calls_snapshot_download(self) -> None:
        """When model is NOT cached, snapshot_download is called with model name."""
        from voiceflow.stt import ensure_sensevoice_downloaded

        mock_snapshot = MagicMock()
        mock_ms_module = MagicMock()
        mock_ms_module.snapshot_download = mock_snapshot

        with (
            patch("voiceflow.stt._is_sensevoice_cached", return_value=False),
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
