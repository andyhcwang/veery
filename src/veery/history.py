"""Local dictation history: audio + transcripts + user-correction labels.

Every dictation is archived (WAV + JSONL record) so accuracy can be measured
and optimized against the user's real voice. When the user edits pasted text
or re-dictates a correction, the corrected text is attached to the matching
record — turning everyday usage into labeled evaluation data (the Wispr Flow
history/divergence flywheel, kept fully local).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veery.config import PROJECT_ROOT, HistoryConfig

logger = logging.getLogger(__name__)


@dataclass
class HistoryRecord:
    record_id: str
    wav_path: Path
    raw_text: str
    final_text: str
    corrected_text: str | None = None


class DictationHistory:
    """Append-only archive with a record cap. All writes swallow errors:
    history must never break dictation."""

    def __init__(self, config: HistoryConfig) -> None:
        base = Path(config.dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        self._dir = base
        self._records_path = base / "records.jsonl"
        self._max_records = max(10, config.max_records)
        self._lock = threading.Lock()
        # Monotonic per-process counter: wall-clock components alone can
        # collide for rapid saves and silently overwrite WAV files.
        self._seq = 0

    def save(self, audio: np.ndarray, sample_rate: int, raw_text: str, final_text: str) -> None:
        """Archive one dictation (audio + texts). Never raises."""
        try:
            import soundfile as sf

            with self._lock:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._seq += 1
                record_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._seq:04d}"
                wav_path = self._dir / f"{record_id}.wav"
                sf.write(wav_path, audio, sample_rate)
                entry = {
                    "id": record_id,
                    "wav": wav_path.name,
                    "raw_text": raw_text,
                    "final_text": final_text,
                    "ts": time.time(),
                }
                with open(self._records_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._enforce_cap_locked()
        except Exception:
            logger.warning("Could not archive dictation history", exc_info=True)

    def log_correction(self, original_text: str, corrected_text: str) -> None:
        """Attach a user correction to the most recent record with matching text.

        Called when the user manually edits pasted text or re-dictates a
        correction — the corrected version becomes the ground-truth label.
        """
        if not original_text or not corrected_text:
            return
        if original_text.strip() == corrected_text.strip():
            return
        try:
            with self._lock:
                records = self._load_locked()
                for entry in reversed(records):
                    if entry.get("final_text") == original_text and not entry.get("corrected_text"):
                        entry["corrected_text"] = corrected_text
                        self._write_all_locked(records)
                        logger.info("History: labeled record %s with user correction", entry["id"])
                        return
        except Exception:
            logger.warning("Could not label history record", exc_info=True)

    def load_records(self) -> list[HistoryRecord]:
        """All records whose WAV file still exists (newest last)."""
        with self._lock:
            entries = self._load_locked()
        out: list[HistoryRecord] = []
        for e in entries:
            wav = self._dir / e.get("wav", "")
            if not wav.is_file():
                continue
            out.append(
                HistoryRecord(
                    record_id=e.get("id", ""),
                    wav_path=wav,
                    raw_text=e.get("raw_text", ""),
                    final_text=e.get("final_text", ""),
                    corrected_text=e.get("corrected_text"),
                )
            )
        return out

    # ------------------------------------------------------------------

    def _load_locked(self) -> list[dict]:
        if not self._records_path.is_file():
            return []
        entries: list[dict] = []
        with open(self._records_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
        return entries

    def _write_all_locked(self, entries: list[dict]) -> None:
        with open(self._records_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def _enforce_cap_locked(self) -> None:
        entries = self._load_locked()
        if len(entries) <= self._max_records:
            return
        drop, keep = entries[: -self._max_records], entries[-self._max_records :]
        for e in drop:
            try:
                (self._dir / e.get("wav", "")).unlink(missing_ok=True)
            except Exception:
                pass
        self._write_all_locked(keep)
