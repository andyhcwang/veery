"""Tests for streaming dictation: segmentation, session worker, splicing."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from veery.app import _splice_texts, _StreamingSession
from veery.audio import AudioRecorder
from veery.config import AudioConfig, StreamingConfig, VADConfig
from veery.stt import STTError


def _make_recorder(streaming: StreamingConfig | None = None) -> AudioRecorder:
    with patch("veery.audio.sd"):
        rec = AudioRecorder(AudioConfig(), VADConfig())
    if streaming is not None:
        rec.configure_streaming(streaming)
    return rec


def _speech_chunk() -> np.ndarray:
    return np.full(512, 0.1, dtype=np.float32)


def _drive(rec: AudioRecorder, prob: float, n: int) -> None:
    """Feed n VAD chunks with a fixed speech probability."""
    rec._vad_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=prob)))
    for _ in range(n):
        rec._process_vad_chunk(_speech_chunk())


STREAM_CFG = StreamingConfig(enabled=True)

# Chunk counts at 32ms/chunk for the default config
PAUSE_CHUNKS = int(0.7 * 1000 / 32)  # 21
MIN_SPEECH_CHUNKS = int(1.0 * 1000 / 32)  # 31
OVERLAP_CHUNKS = int(200 / 32)  # 6


class TestStreamingSegmentation:
    def _manual_recorder(self, cfg: StreamingConfig = STREAM_CFG) -> tuple[AudioRecorder, list]:
        rec = _make_recorder(cfg)
        events: list = []
        rec.set_finalize_callback(lambda seq, chunks, end: events.append((seq, chunks, end)))
        with patch.object(rec, "_ensure_vad_loaded"), patch("veery.audio.sd"):
            rec.prepare_stream(manual_mode=True)
        # prepare_stream resets the callback? It must NOT — verify separately.
        return rec, events

    def test_pause_after_speech_emits_one_segment(self) -> None:
        rec, events = self._manual_recorder()
        _drive(rec, 0.9, 40)  # 40 speech chunks > min 31
        _drive(rec, 0.0, PAUSE_CHUNKS)  # exactly the finalize pause
        assert len(events) == 1
        seq, chunks, end = events[0]
        assert seq == 0
        assert end == 40 + PAUSE_CHUNKS  # buffer length at cut
        assert len(chunks) == end  # first segment has no earlier overlap
        # Continued silence up to just below recording-end must not re-emit
        _drive(rec, 0.0, 10)
        assert len(events) == 1

    def test_short_speech_does_not_emit(self) -> None:
        rec, events = self._manual_recorder()
        _drive(rec, 0.9, 10)  # only ~0.3s of speech
        _drive(rec, 0.0, PAUSE_CHUNKS + 5)
        assert events == []

    def test_second_segment_includes_overlap_and_advances_cursor(self) -> None:
        rec, events = self._manual_recorder()
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        _drive(rec, 0.9, 40)  # speech resumes
        _drive(rec, 0.0, PAUSE_CHUNKS)
        assert len(events) == 2
        _, chunks1, end1 = events[0]
        seq2, chunks2, end2 = events[1]
        assert seq2 == 1
        assert end2 > end1
        assert len(chunks2) == (end2 - end1) + OVERLAP_CHUNKS

    def test_max_segment_cap_forces_cut_without_pause(self) -> None:
        cfg = StreamingConfig(enabled=True, max_segment_sec=2.0)
        rec, events = self._manual_recorder(cfg)
        _drive(rec, 0.9, 80)  # 2.56s of continuous speech > 2.0s cap
        assert len(events) >= 1
        assert events[0][0] == 0

    def test_wait_mode_never_emits(self) -> None:
        rec = _make_recorder(STREAM_CFG)
        events: list = []
        rec.set_finalize_callback(lambda *a: events.append(a))
        with patch.object(rec, "_ensure_vad_loaded"), patch("veery.audio.sd"):
            rec.prepare_stream(manual_mode=False)
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        assert events == []

    def test_disabled_config_never_emits(self) -> None:
        rec, events = self._manual_recorder(StreamingConfig(enabled=False))
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        assert events == []

    def test_build_tail_slices_from_committed_chunk(self) -> None:
        rec, events = self._manual_recorder()
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        _drive(rec, 0.9, 40)  # tail speech after the cut
        (seq, chunks, end) = events[0]
        tail = rec.build_tail(end)
        assert tail is not None
        assert tail.vad_confirmed is True
        # tail = overlap + chunks after the cut
        expected_chunks = OVERLAP_CHUNKS + 40
        assert len(tail.audio) == expected_chunks * 512

    def test_build_tail_zero_falls_back_to_batch_segment(self) -> None:
        rec, _ = self._manual_recorder()
        _drive(rec, 0.9, 40)
        with patch.object(rec, "_build_segment", return_value="sentinel") as bs:
            assert rec.build_tail(0) == "sentinel"
        bs.assert_called_once_with(use_raw=True)

    def test_stop_capture_clears_callback_and_signals(self) -> None:
        rec, events = self._manual_recorder()
        with patch("veery.audio.threading.Thread"):
            rec.stop_capture()
        assert rec._finalize_cb is None
        assert rec._done_event.is_set()
        assert rec._manual_stop_event.is_set()
        # further chunks emit nothing
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        assert events == []

    def test_prepare_stream_resets_cursor_but_keeps_callback(self) -> None:
        rec, events = self._manual_recorder()
        _drive(rec, 0.9, 40)
        _drive(rec, 0.0, PAUSE_CHUNKS)
        assert len(events) == 1
        assert rec._finalized_chunks > 0
        rec._stream = None  # previous recording's stream was closed
        with patch.object(rec, "_ensure_vad_loaded"), patch("veery.audio.sd"):
            rec.prepare_stream(manual_mode=True)
        assert rec._finalized_chunks == 0
        assert rec._segment_seq == 0


class TestStreamingSession:
    def _stt(self, texts: dict[int, str | Exception]) -> MagicMock:
        stt = MagicMock()
        calls = {"n": 0}

        def transcribe(audio, sr):
            result = texts.get(calls["n"], "")
            calls["n"] += 1
            if isinstance(result, Exception):
                raise result
            return result

        stt.transcribe.side_effect = transcribe
        return stt

    def test_segments_commit_in_order(self) -> None:
        stt = self._stt({0: "hello", 1: "world"})
        session = _StreamingSession(stt, 16000)
        session.enqueue(0, [np.zeros(512, dtype=np.float32)], 10)
        session.enqueue(1, [np.zeros(512, dtype=np.float32)], 20)
        assert session.drain(5.0)
        parts, end = session.committed()
        assert parts == ["hello", "world"]
        assert end == 20

    def test_failed_segment_stops_committed_prefix(self) -> None:
        stt = self._stt({0: "hello", 1: STTError("boom"), 2: "after"})
        session = _StreamingSession(stt, 16000)
        for seq in range(3):
            session.enqueue(seq, [np.zeros(512, dtype=np.float32)], (seq + 1) * 10)
        assert session.drain(5.0)
        parts, end = session.committed()
        assert parts == ["hello"]
        assert end == 10  # tail re-covers everything from the failed segment on

    def test_empty_text_segment_still_commits(self) -> None:
        stt = self._stt({0: "", 1: "world"})
        session = _StreamingSession(stt, 16000)
        session.enqueue(0, [np.zeros(512, dtype=np.float32)], 10)
        session.enqueue(1, [np.zeros(512, dtype=np.float32)], 20)
        assert session.drain(5.0)
        parts, end = session.committed()
        assert parts == ["", "world"]
        assert end == 20

    def test_hallucinated_segment_commits_empty(self) -> None:
        stt = self._stt({0: "why why why why why why why why why why"})
        session = _StreamingSession(stt, 16000)
        session.enqueue(0, [np.zeros(512, dtype=np.float32)], 10)
        assert session.drain(5.0)
        parts, end = session.committed()
        assert parts == [""]
        assert end == 10

    def test_drain_times_out_when_worker_hangs(self) -> None:
        release = threading.Event()
        stt = MagicMock()
        stt.transcribe.side_effect = lambda a, sr: release.wait(10) or "late"
        session = _StreamingSession(stt, 16000)
        session.enqueue(0, [np.zeros(512, dtype=np.float32)], 10)
        time.sleep(0.05)  # let the worker enter transcribe
        assert session.drain(0.2) is False
        parts, end = session.committed()
        assert parts == []
        assert end == 0
        release.set()


class TestSpliceTexts:
    def test_english_parts_get_spaces(self) -> None:
        assert _splice_texts(["Hello there", "how are you"]) == "Hello there how are you"

    def test_cjk_parts_join_without_space(self) -> None:
        assert _splice_texts(["你好世界", "今天天气不错"]) == "你好世界今天天气不错"

    def test_mixed_cjk_english_boundaries(self) -> None:
        assert _splice_texts(["让Claude帮我", "review这个PR"]) == "让Claude帮我review这个PR"
        assert _splice_texts(["check the latency", "然后发给我"]) == "check the latency然后发给我"

    def test_punctuation_lead_gets_no_space(self) -> None:
        assert _splice_texts(["Hello", ", world"]) == "Hello, world"

    def test_empty_parts_skipped(self) -> None:
        assert _splice_texts(["", "  ", "hello", "", "world"]) == "hello world"

    def test_all_empty_returns_empty(self) -> None:
        assert _splice_texts(["", "  "]) == ""
