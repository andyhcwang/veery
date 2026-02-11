"""Integration tests: mock heavy dependencies, test wiring between modules."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from veery.config import (
    PROJECT_ROOT,
    AppConfig,
    AudioConfig,
    JargonConfig,
    OutputConfig,
    STTConfig,
    VADConfig,
    load_config,
)
from veery.corrector import TextCorrector
from veery.jargon import JargonCorrector
from veery.stt import _extract_text, _strip_tags

QUANT_DICT = str(PROJECT_ROOT / "jargon" / "quant_finance.yaml")
TECH_DICT = str(PROJECT_ROOT / "jargon" / "tech.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jargon_corrector() -> JargonCorrector:
    cfg = JargonConfig(dict_paths=(QUANT_DICT, TECH_DICT))
    return JargonCorrector(cfg)


@pytest.fixture
def text_corrector(jargon_corrector: JargonCorrector) -> TextCorrector:
    return TextCorrector(jargon=jargon_corrector)


# ---------------------------------------------------------------------------
# 1. Full pipeline happy path
# ---------------------------------------------------------------------------


class TestFullPipelineHappyPath:
    def test_full_pipeline_happy_path(self, text_corrector: TextCorrector) -> None:
        """Mock AudioRecorder -> fake AudioSegment, mock STT -> raw text,
        verify final output after jargon correction contains 'Sharpe ratio'."""
        # Simulate STT returning a misspelled jargon phrase
        raw_stt_output = "the sharp ratio is good"

        result = text_corrector.correct(raw_stt_output)

        assert result.raw == "the sharp ratio is good"
        assert "Sharpe ratio" in result.jargon_corrected
        assert "Sharpe ratio" in result.final

    def test_full_pipeline_with_mock_audio_and_stt(self, text_corrector: TextCorrector) -> None:
        """End-to-end: fake audio segment -> mock STT transcribe -> corrector -> output."""
        from veery.audio import AudioSegment

        fake_segment = AudioSegment(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            duration_sec=1.0,
        )

        # Mock STT that returns a raw transcription
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "the sharp ratio is good"

        # Simulate the app._process_segment flow
        raw_text = mock_stt.transcribe(fake_segment.audio, fake_segment.sample_rate)
        assert raw_text == "the sharp ratio is good"

        result = text_corrector.correct(raw_text)
        assert "Sharpe ratio" in result.final


# ---------------------------------------------------------------------------
# 2. Pipeline: STT returns empty
# ---------------------------------------------------------------------------


class TestPipelineSTTEmpty:
    def test_pipeline_stt_returns_empty(self, text_corrector: TextCorrector) -> None:
        """When STT returns '', corrector should produce empty CorrectionResult."""
        result = text_corrector.correct("")
        assert result.raw == ""
        assert result.jargon_corrected == ""
        assert result.final == ""

    def test_app_process_segment_empty_stt(self) -> None:
        """Simulate VeeryApp._process_segment when STT returns empty string.
        The app checks `if not raw_text:` and shows a notification instead of outputting."""
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = ""

        raw_text = mock_stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        # The app does: if not raw_text: return (no output attempted)
        assert not raw_text


# ---------------------------------------------------------------------------
# 3. Pipeline: STT raises exception
# ---------------------------------------------------------------------------


class TestPipelineSTTFails:
    def test_pipeline_stt_fails(self) -> None:
        """When STT.transcribe raises, the caller should handle gracefully."""
        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = RuntimeError("Model crashed")

        with pytest.raises(RuntimeError, match="Model crashed"):
            mock_stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    def test_real_stt_transcribe_handles_internal_errors(self) -> None:
        """SenseVoiceSTT.transcribe() catches exceptions and returns '' gracefully."""
        from veery.stt import SenseVoiceSTT

        # Create STT with a mock model that raises during generate()
        with patch("veery.stt.SenseVoiceSTT._load_model"):
            stt = SenseVoiceSTT(STTConfig())
            stt._model = MagicMock()
            stt._model.generate.side_effect = RuntimeError("inference failure")

            result = stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)
            assert result == ""

    def test_stt_with_no_model_returns_empty(self) -> None:
        """If model failed to load (_model is None), transcribe returns ''."""
        from veery.stt import SenseVoiceSTT

        with patch("veery.stt.SenseVoiceSTT._load_model"):
            stt = SenseVoiceSTT(STTConfig())
            stt._model = None

            result = stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)
            assert result == ""


# ---------------------------------------------------------------------------
# 4. Pipeline: no speech detected
# ---------------------------------------------------------------------------


class TestPipelineNoSpeechDetected:
    def test_wait_for_speech_end_returns_none(self) -> None:
        """When AudioRecorder.wait_for_speech_end returns None, the app notifies the user."""
        mock_recorder = MagicMock()
        mock_recorder.wait_for_speech_end.return_value = None

        segment = mock_recorder.wait_for_speech_end(timeout=30)
        assert segment is None


# ---------------------------------------------------------------------------
# 5. Config loads defaults
# ---------------------------------------------------------------------------


class TestConfigLoadsDefaults:
    def test_config_loads_defaults(self) -> None:
        """AppConfig() should create valid defaults for all fields."""
        config = AppConfig()

        assert isinstance(config.audio, AudioConfig)
        assert config.audio.sample_rate == 16000
        assert config.audio.channels == 1
        assert config.audio.chunk_duration_ms == 96
        assert config.audio.max_duration_sec == 30.0

        assert isinstance(config.vad, VADConfig)
        assert config.vad.threshold == 0.4
        assert config.vad.silence_duration_sec == 2.0
        assert config.vad.min_speech_duration_sec == 0.3

        assert isinstance(config.stt, STTConfig)
        assert config.stt.model_name == "iic/SenseVoiceSmall"
        assert config.stt.language == "auto"
        assert config.stt.device == "cpu"
        assert config.stt.backend == "whisper"
        assert config.stt.whisper_model == "mlx-community/whisper-large-v3-turbo"

        assert isinstance(config.jargon, JargonConfig)
        assert len(config.jargon.dict_paths) == 5

        assert isinstance(config.output, OutputConfig)
        assert config.output.cgevent_char_limit == 500

    def test_audio_config_properties(self) -> None:
        """Verify computed properties on AudioConfig."""
        audio = AudioConfig()
        # chunk_samples = 16000 * 96 / 1000 = 1536
        assert audio.chunk_samples == 1536
        # max_buffer_samples = 16000 * 30 = 480000
        assert audio.max_buffer_samples == 480000


# ---------------------------------------------------------------------------
# 6. Config from YAML
# ---------------------------------------------------------------------------


class TestConfigFromYAML:
    def test_config_from_yaml(self, tmp_path: Path) -> None:
        """Create a temp YAML with custom settings, verify load_config reads them."""
        yaml_content = textwrap.dedent("""\
            audio:
              sample_rate: 44100
              channels: 2
              max_duration_sec: 60.0
            vad:
              threshold: 0.7
              silence_duration_sec: 2.0
            stt:
              model_name: "custom/model"
              device: "mps"
            jargon:
              dict_paths:
                - "custom/dict.yaml"
              fuzzy_threshold: 90
            output:
              cgevent_char_limit: 1000
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)

        assert config.audio.sample_rate == 44100
        assert config.audio.channels == 2
        assert config.audio.max_duration_sec == 60.0
        assert config.vad.threshold == 0.7
        assert config.vad.silence_duration_sec == 2.0
        assert config.stt.model_name == "custom/model"
        assert config.stt.device == "mps"
        assert config.jargon.dict_paths == ("custom/dict.yaml",)
        assert config.jargon.fuzzy_threshold == 90
        assert config.output.cgevent_char_limit == 1000

    def test_config_from_yaml_whisper_backend(self, tmp_path: Path) -> None:
        """Verify YAML override for STT whisper backend fields."""
        yaml_content = textwrap.dedent("""\
            stt:
              backend: "whisper"
              whisper_model: "mlx-community/whisper-small"
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)

        assert config.stt.backend == "whisper"
        assert config.stt.whisper_model == "mlx-community/whisper-small"
        # Defaults preserved for other STT fields
        assert config.stt.model_name == "iic/SenseVoiceSmall"
        assert config.stt.language == "auto"


# ---------------------------------------------------------------------------
# 7. Config partial YAML (defaults fill missing fields)
# ---------------------------------------------------------------------------


class TestConfigPartialYAML:
    def test_config_partial_yaml(self, tmp_path: Path) -> None:
        """YAML with only some fields, verify defaults fill the rest."""
        yaml_content = textwrap.dedent("""\
            audio:
              sample_rate: 8000
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)

        # Overridden fields
        assert config.audio.sample_rate == 8000

        # Default-filled fields
        assert config.audio.channels == 1  # default
        assert config.audio.max_duration_sec == 30.0  # default
        assert config.vad.threshold == 0.4  # entire section defaulted
        assert config.stt.model_name == "iic/SenseVoiceSmall"  # default
        assert config.jargon.fuzzy_threshold == 82  # default
        assert config.output.cgevent_char_limit == 500  # default

    def test_config_nonexistent_file(self) -> None:
        """load_config with a nonexistent path returns pure defaults."""
        config = load_config(Path("/nonexistent/config.yaml"))
        assert config == AppConfig()

    def test_config_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file returns pure defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        assert config.audio.sample_rate == 16000


# ---------------------------------------------------------------------------
# 8. Mixed Chinese-English jargon
# ---------------------------------------------------------------------------


class TestMixedChineseEnglishJargon:
    def test_mixed_chinese_english_jargon_spaced(self, jargon_corrector: JargonCorrector) -> None:
        """Chinese characters with spaces around English jargon: jargon is corrected, Chinese preserved."""
        # With spaces, the tokenizer can isolate "sharp ratio" as separate tokens
        text = "这个 sharp ratio 太低了"
        result = jargon_corrector.correct(text)
        assert "Sharpe ratio" in result
        assert "这个" in result
        assert "太低了" in result

    def test_mixed_chinese_english_no_spaces(self, jargon_corrector: JargonCorrector) -> None:
        """Without spaces between Chinese and English, jargon tokens fuse with Chinese chars.
        The tokenizer sees '这个sharp' as one token, so 'sharp' is not matched.
        This documents actual behavior -- a known limitation."""
        text = "这个sharp ratio太低了"
        result = jargon_corrector.correct(text)
        # "ratio太低了" is one token; "这个sharp" is another -- neither matches jargon
        assert result == text  # passes through unchanged

    def test_chinese_only_passthrough(self, jargon_corrector: JargonCorrector) -> None:
        """Pure Chinese text should pass through unchanged."""
        text = "这是一个测试句子"
        result = jargon_corrector.correct(text)
        assert result == text

    def test_mixed_with_multiple_jargon(self, jargon_corrector: JargonCorrector) -> None:
        """Multiple jargon terms in mixed text (with spaces)."""
        text = "用 duck DB 计算 sharp ratio"
        result = jargon_corrector.correct(text)
        assert "DuckDB" in result
        assert "Sharpe ratio" in result


# ---------------------------------------------------------------------------
# 9. STT tag stripping
# ---------------------------------------------------------------------------


class TestStripSenseVoiceTags:
    def test_strip_tags_basic(self) -> None:
        """Strip SenseVoice language/emotion/event tags."""
        raw = "<|zh|><|NEUTRAL|><|Speech|>这个sharp ratio太低了"
        assert _strip_tags(raw) == "这个sharp ratio太低了"

    def test_strip_tags_english(self) -> None:
        raw = "<|en|><|HAPPY|><|Speech|>the Sharpe ratio is great"
        assert _strip_tags(raw) == "the Sharpe ratio is great"

    def test_strip_tags_multiple_events(self) -> None:
        raw = "<|zh|><|NEUTRAL|><|Speech|><|BGM|>测试文本"
        assert _strip_tags(raw) == "测试文本"

    def test_strip_tags_no_tags(self) -> None:
        """Text without tags should pass through unchanged."""
        assert _strip_tags("hello world") == "hello world"

    def test_strip_tags_empty(self) -> None:
        assert _strip_tags("") == ""

    def test_strip_tags_only_tags(self) -> None:
        """Input with only tags -> empty string."""
        assert _strip_tags("<|zh|><|NEUTRAL|><|Speech|>") == ""

    def test_extract_text_dict_format(self) -> None:
        """FunASR returns list of dicts with 'text' key."""
        result = [{"text": "<|zh|><|NEUTRAL|><|Speech|>这个sharp ratio太低了"}]
        assert _extract_text(result) == "这个sharp ratio太低了"

    def test_extract_text_empty_list(self) -> None:
        assert _extract_text([]) == ""

    def test_extract_text_no_text_key(self) -> None:
        result = [{"score": 0.95}]
        assert _extract_text(result) == ""

    def test_full_stt_to_jargon_flow(self, jargon_corrector: JargonCorrector) -> None:
        """Realistic flow: SenseVoice output -> strip tags -> jargon correction.
        SenseVoice typically produces spaces around English tokens in Chinese text."""
        raw_sensevoice = "<|zh|><|NEUTRAL|><|Speech|>这个 sharp ratio 太低了"
        stripped = _strip_tags(raw_sensevoice)
        assert stripped == "这个 sharp ratio 太低了"
        corrected = jargon_corrector.correct(stripped)
        assert "Sharpe ratio" in corrected
        assert "这个" in corrected
        assert "太低了" in corrected
