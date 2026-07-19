#!/usr/bin/env python
"""Offline accuracy grid over archived dictation history.

Re-transcribes labeled history WAVs under config variants and ranks them by
mixed CN/EN error rate. Read-only: prints a leaderboard and writes a report
to history/optimize-report.{md,json}; applying a winner is a human (or
scheduled-agent) decision.

Exit codes: 0 = report written, 2 = not enough labeled data yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veery.config import STTConfig, load_config  # noqa: E402
from veery.evaluate import (  # noqa: E402
    _parse_script,
    evaluate_pairs,
    match_script_line,
)
from veery.history import DictationHistory, HistoryRecord  # noqa: E402

# Config-level variants to try against "current". Variants identical to the
# current config are skipped automatically.
VARIANTS: list[tuple[str, dict]] = [
    ("current", {}),
    ("backend=sensevoice", {"backend": "sensevoice"}),
    ("language=zh", {"language": "zh"}),
    ("jargon-prompt=off", {"whisper_use_jargon_prompt": False}),
    ("prompt=64x400", {"whisper_prompt_terms_limit": 64, "whisper_prompt_char_limit": 400}),
]


def collect_labeled(
    records: list[HistoryRecord], script_path: str | None
) -> list[tuple[HistoryRecord, str, str]]:
    """(record, reference, category) for every record with a usable label."""
    script_pairs = _parse_script(script_path) if script_path else []
    lines = [line for _, line in script_pairs]
    category_of = {line: cat for cat, line in script_pairs}

    labeled: list[tuple[HistoryRecord, str, str]] = []
    for r in records:
        if r.corrected_text:
            labeled.append((r, r.corrected_text, "correction"))
            continue
        if lines:
            best = match_script_line(r.final_text, lines)
            if best is not None:
                labeled.append((r, best[0], category_of[best[0]]))
    return labeled


def _build_jargon_prompt(corrector, stt_cfg: STTConfig, tracker) -> str | None:
    """Mirror of VeeryApp._build_whisper_jargon_prompt for headless use."""
    if not stt_cfg.whisper_use_jargon_prompt:
        return None
    terms = list(corrector.jargon.dictionary.canonical_terms)
    if not terms:
        return None
    if tracker is not None:
        terms = tracker.rank(terms)
    prefix = "Technical dictation. Prefer these exact spellings: "
    selected: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue
        if len(prefix + ", ".join(selected + [cleaned]) + ".") > stt_cfg.whisper_prompt_char_limit:
            break
        selected.append(cleaned)
        if len(selected) >= stt_cfg.whisper_prompt_terms_limit:
            break
    return prefix + ", ".join(selected) + "." if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default=None, help="Benchmark reading script path")
    parser.add_argument("--min-labels", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    history = DictationHistory(cfg.history)
    records = history.load_records()
    labeled = collect_labeled(records, args.script)
    print(f"History: {len(records)} records, {len(labeled)} labeled")
    if len(labeled) < args.min_labels:
        print(f"SKIP: need at least {args.min_labels} labeled records to optimize.")
        sys.exit(2)

    import soundfile as sf

    from veery.corrector import TextCorrector
    from veery.jargon import JargonCorrector, JargonUsageTracker
    from veery.stt import WhisperSTT, create_stt

    corrector = TextCorrector(JargonCorrector(cfg.jargon))
    try:
        stats_path = Path(cfg.jargon.learned_path or "jargon/learned.yaml")
        if not stats_path.is_absolute():
            stats_path = ROOT / stats_path
        tracker = JargonUsageTracker(stats_path.parent / "usage_stats.yaml")
    except Exception:
        tracker = None

    clips: list[tuple[object, int, str, str]] = []  # audio, sr, ref, category
    for rec, ref, cat in labeled:
        audio, sr = sf.read(rec.wav_path, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        clips.append((audio, sr, ref, cat))

    report: list[dict] = []
    for name, overrides in VARIANTS:
        stt_cfg = replace(cfg.stt, **overrides) if overrides else cfg.stt
        if overrides and stt_cfg == cfg.stt:
            print(f"{name:<24} (identical to current, skipped)")
            continue
        try:
            stt = create_stt(stt_cfg)
        except Exception as exc:
            print(f"{name:<24} LOAD FAILED: {exc}")
            continue
        if isinstance(stt, WhisperSTT):
            try:
                stt.set_runtime_hints(prompt=_build_jargon_prompt(corrector, stt_cfg, tracker))
            except Exception:
                pass

        pairs: list[tuple[str, str]] = []
        per_cat: dict[str, list[tuple[str, str]]] = {}
        t0 = time.time()
        for audio, sr, ref, cat in clips:
            try:
                hyp_raw = stt.transcribe(audio, sr)
            except Exception:
                hyp_raw = ""
            hyp = corrector.correct(hyp_raw).final
            pairs.append((hyp, ref))
            per_cat.setdefault(cat, []).append((hyp, ref))
        result = evaluate_pairs(pairs, name)
        categories = {c: evaluate_pairs(p, c).error_rate for c, p in per_cat.items()}
        print(f"{result}   ({time.time() - t0:.0f}s)")
        report.append({
            "variant": name,
            "overrides": overrides,
            "mer": result.error_rate,
            "n": result.n_samples,
            "categories": categories,
        })
        try:
            stt.release_resources()
        except Exception:
            pass

    if not report:
        print("No variants produced results.")
        sys.exit(1)

    report.sort(key=lambda r: r["mer"])
    current = next((r for r in report if r["variant"] == "current"), None)
    best = report[0]

    out_dir = ROOT / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "optimize-report.json").write_text(
        json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "results": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [f"# Veery accuracy grid — {time.strftime('%Y-%m-%d %H:%M')}",
          f"Labeled samples: {len(clips)}", "", "| variant | MER | per-category |", "|---|---|---|"]
    for r in report:
        cats = ", ".join(f"{c}: {v * 100:.1f}%" for c, v in sorted(r["categories"].items()))
        md.append(f"| {r['variant']} | {r['mer'] * 100:.2f}% | {cats} |")
    if current is not None and best["variant"] != "current" and current["mer"] > 0:
        rel = (current["mer"] - best["mer"]) / current["mer"] * 100
        md.append(f"\nBest: **{best['variant']}** — {rel:.0f}% relative improvement over current.")
        md.append(f"Apply via config.yaml stt overrides: `{best['overrides']}`")
    else:
        md.append("\nCurrent config is already the best variant.")
    (out_dir / "optimize-report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nReport: {out_dir / 'optimize-report.md'}")


if __name__ == "__main__":
    main()
