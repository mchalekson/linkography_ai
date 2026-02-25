#!/usr/bin/env python
"""Compare time-based vs index-based thirds for CDP entropy.

Computes CDP score entropy using:
  1) Index-based thirds (utterance count)
  2) Time-based thirds (meeting duration)

Outputs a per-session comparison table and a summary report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from linkography_ai.discovery import list_conferences
from linkography_ai.entropy import shannon_entropy_from_counts
from linkography_ai.segmentation import segment_thirds

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def time_str_to_sec(s: Any) -> float:
    if not isinstance(s, str) or ":" not in s:
        return math.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return math.nan
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return math.nan


def extract_cdp_score(u: Dict[str, Any]) -> list[str]:
    ad = u.get("annotation_dict") or u.get("annotations") or {}
    if isinstance(ad, dict):
        cdp_data = ad.get("Coordination and Decision Practices")
        if isinstance(cdp_data, dict):
            score = cdp_data.get("score")
            if score is not None:
                return [f"CDP_score_{score}"]
    return []


def load_session(session_path: Path) -> list[dict]:
    obj = json.loads(session_path.read_text())
    if isinstance(obj, dict) and isinstance(obj.get("all_data"), list):
        return obj["all_data"]
    if isinstance(obj, dict) and isinstance(obj.get("utterances"), list):
        return obj["utterances"]
    if isinstance(obj, list):
        return obj
    return []


def assign_time_third(t_sec: float, start: float, end: float) -> str:
    total = max(1.0, end - start)
    one_third = start + total / 3.0
    two_third = start + 2.0 * total / 3.0
    if t_sec < one_third:
        return "beginning"
    if t_sec < two_third:
        return "middle"
    return "end"


def compute_entropy_from_counts(counts: Dict[str, int], normalize: bool) -> float:
    return shannon_entropy_from_counts(list(counts.values()), normalize=normalize) if counts else float("nan")


def analyze_session(session_path: Path, normalize: bool) -> Optional[dict]:
    utter_list = load_session(session_path)
    if not utter_list:
        return None

    # index-based thirds
    index_labels = segment_thirds(len(utter_list))
    index_counts = {"beginning": {}, "middle": {}, "end": {}}

    # time-based thirds
    times = []
    for u in utter_list:
        start_time = u.get("start_time")
        start_sec = time_str_to_sec(start_time)
        times.append(start_sec)
    times = [t for t in times if not math.isnan(t)]
    if not times:
        return None
    t_start, t_end = min(times), max(times)

    time_counts = {"beginning": {}, "middle": {}, "end": {}}

    for u, label in zip(utter_list, index_labels):
        scores = extract_cdp_score(u)
        for s in scores:
            index_counts[label][s] = index_counts[label].get(s, 0) + 1

    for u in utter_list:
        start_sec = time_str_to_sec(u.get("start_time"))
        if math.isnan(start_sec):
            continue
        t_label = assign_time_third(start_sec, t_start, t_end)
        scores = extract_cdp_score(u)
        for s in scores:
            time_counts[t_label][s] = time_counts[t_label].get(s, 0) + 1

    row = {"session_id": session_path.stem}
    for seg in ["beginning", "middle", "end"]:
        row[f"entropy_index_{seg}"] = compute_entropy_from_counts(index_counts[seg], normalize)
        row[f"entropy_time_{seg}"] = compute_entropy_from_counts(time_counts[seg], normalize)
        row[f"n_cdp_index_{seg}"] = int(sum(index_counts[seg].values()))
        row[f"n_cdp_time_{seg}"] = int(sum(time_counts[seg].values()))
        row[f"entropy_diff_{seg}"] = row[f"entropy_time_{seg}"] - row[f"entropy_index_{seg}"]

    return row


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("TIME-BASED vs INDEX-BASED THIRDS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(df)}\n\n")

        for seg in ["beginning", "middle", "end"]:
            diff = df[f"entropy_diff_{seg}"].dropna()
            f.write(f"{seg.upper()}\n")
            f.write(f"  Mean diff (time - index): {diff.mean():.4f}\n")
            f.write(f"  Median diff: {diff.median():.4f}\n")
            f.write(f"  Std diff: {diff.std(ddof=1):.4f}\n")
            f.write("\n")

        f.write("CORRELATIONS (time vs index)\n")
        f.write("-" * 80 + "\n")
        for seg in ["beginning", "middle", "end"]:
            a = df[f"entropy_time_{seg}"]
            b = df[f"entropy_index_{seg}"]
            valid = df[[f"entropy_time_{seg}", f"entropy_index_{seg}"]].dropna()
            r = valid.corr().iloc[0, 1] if len(valid) > 1 else float("nan")
            f.write(f"  {seg}: r = {r:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare time-based vs index-based thirds")
    parser.add_argument("--conference", default="ALL", help="Conference code or ALL")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences() if args.conference.upper() == "ALL" else [args.conference]
    rows = []

    for conf in conferences:
        session_dir = DATA_DIR / conf / "session_data"
        if not session_dir.exists():
            continue
        for session_path in sorted(session_dir.glob("*.json")):
            row = analyze_session(session_path, normalize=args.normalize)
            if row is None:
                continue
            row["conference"] = conf
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "time_binning_comparison.csv"
    df.to_csv(out_path, index=False)

    summary_path = ANALYSIS_DIR / "time_binning_comparison_summary.txt"
    write_summary(df, summary_path)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
