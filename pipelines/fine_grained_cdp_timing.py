#!/usr/bin/env python
"""Fine-Grained CDP Timing (5-10 min bins)

Computes CDP entropy in smaller time windows to detect inflection points.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linkography_ai.discovery import list_conferences
from linkography_ai.entropy import shannon_entropy_from_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"
FIGURES_DIR = REPO_ROOT / "figures" / "final"


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


def extract_cdp_score(u: dict) -> Optional[str]:
    ann = u.get("annotations") or u.get("annotation_dict") or {}
    if not isinstance(ann, dict):
        return None
    cdp_data = ann.get("Coordination and Decision Practices")
    if isinstance(cdp_data, dict):
        score = cdp_data.get("score")
        if score is not None:
            return f"score_{score}"
    return None


def analyze_session(session_path: Path, bin_sec: int = 300) -> Optional[pd.DataFrame]:
    """Compute per-bin entropy for session."""
    try:
        obj = json.loads(session_path.read_text())
    except Exception:
        return None

    session_json = obj if isinstance(obj, dict) and "all_data" in obj else None
    if session_json is None:
        return None

    rows = []
    for u in session_json.get("all_data", []):
        if not isinstance(u, dict):
            continue
        start_time = u.get("start_time")
        start_sec = time_str_to_sec(start_time)
        if math.isnan(start_sec):
            continue
        score = extract_cdp_score(u)
        if score is None:
            continue
        rows.append({"start_sec": start_sec, "score": score})

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    df["bin_sec"] = (df["start_sec"] // bin_sec).astype(int) * bin_sec

    # Compute entropy per bin
    results = []
    for bin_sec_val, group in df.groupby("bin_sec"):
        counts = Counter(group["score"])
        entropy = shannon_entropy_from_counts(list(counts.values()), normalize=True)
        results.append({
            "session_id": session_path.stem,
            "bin_sec": int(bin_sec_val),
            "bin_min": bin_sec_val / 60.0,
            "n_cdp": len(group),
            "entropy": entropy,
        })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-grained CDP timing analysis")
    parser.add_argument("--bin-sec", type=int, default=300, help="Bin width in seconds (default 300 = 5 min)")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences()
    all_rows = []

    for conf in conferences:
        session_dir = DATA_DIR / conf / "session_data"
        if not session_dir.exists():
            continue
        for session_path in sorted(session_dir.glob("*.json")):
            df_session = analyze_session(session_path, bin_sec=args.bin_sec)
            if df_session is None:
                continue
            df_session["conference"] = conf
            all_rows.append(df_session)

    if not all_rows:
        print("No sessions analyzed.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    out_path = TABLES_DIR / f"cdp_fine_grained_entropy_{args.bin_sec}s.csv"
    df.to_csv(out_path, index=False)

    report_path = ANALYSIS_DIR / f"cdp_fine_grained_summary_{args.bin_sec}s.txt"
    with open(report_path, "w") as f:
        f.write(f"CDP FINE-GRAINED ENTROPY ({args.bin_sec}s bins)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total bin-level observations: {len(df)}\n")
        f.write(f"Mean entropy per bin: {df['entropy'].mean():.4f}\n")
        f.write(f"Std entropy: {df['entropy'].std(ddof=1):.4f}\n")
        f.write(f"Entropy range: [{df['entropy'].min():.4f}, {df['entropy'].max():.4f}]\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
