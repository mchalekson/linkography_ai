#!/usr/bin/env python
"""CDP Content Analysis

Extracts utterances by CDP score (1 vs 2) and summarizes their characteristics:
- Count and percentage of score 1 vs score 2
- Sample utterance excerpts
- Token-level statistics
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from linkography_ai.discovery import list_conferences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def extract_cdp_utterances(session_json: dict) -> tuple[list[dict], list[dict]]:
    """Extract utterances with CDP score 1 and score 2."""
    score1 = []
    score2 = []

    for u in session_json.get("all_data", []):
        if not isinstance(u, dict):
            continue
        text = (u.get("transcript") or "").strip()
        if not text:
            continue

        ann = u.get("annotations") or u.get("annotation_dict") or {}
        if not isinstance(ann, dict):
            continue

        cdp_data = ann.get("Coordination and Decision Practices")
        if not isinstance(cdp_data, dict):
            continue

        score = cdp_data.get("score")
        if score == 1:
            score1.append({"text": text})
        elif score == 2:
            score2.append({"text": text})

    return score1, score2


def tokenize(text: str) -> list[str]:
    words = text.lower().split()
    return [w.strip(",.!?;:") for w in words if len(w.strip(",.!?;:")) > 2]


def analyze_session(session_path: Path) -> Optional[dict]:
    try:
        obj = json.loads(session_path.read_text())
    except Exception:
        return None

    session_json = obj
    if isinstance(obj, dict) and "all_data" in obj:
        session_json = obj
    else:
        return None

    score1, score2 = extract_cdp_utterances(session_json)
    if not score1 and not score2:
        return None

    # Sample utterances (first 2)
    sample1 = [u["text"][:100] for u in score1[:2]]
    sample2 = [u["text"][:100] for u in score2[:2]]

    # Token counts
    tokens1 = [t for u in score1 for t in tokenize(u["text"])]
    tokens2 = [t for u in score2 for t in tokenize(u["text"])]

    return {
        "session_id": session_path.stem,
        "n_cdp_score1": len(score1),
        "n_cdp_score2": len(score2),
        "pct_score1": 100 * len(score1) / (len(score1) + len(score2)) if score1 or score2 else math.nan,
        "pct_score2": 100 * len(score2) / (len(score1) + len(score2)) if score1 or score2 else math.nan,
        "mean_tokens_score1": sum(len(tokenize(u["text"])) for u in score1) / len(score1) if score1 else math.nan,
        "mean_tokens_score2": sum(len(tokenize(u["text"])) for u in score2) / len(score2) if score2 else math.nan,
        "sample_score1": " | ".join(sample1) if sample1 else "",
        "sample_score2": " | ".join(sample2) if sample2 else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CDP content analysis")
    parser.add_argument("--conference", default="ALL", help="Conference code or ALL")
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
            row = analyze_session(session_path)
            if row is None:
                continue
            row["conference"] = conf
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "cdp_content_analysis.csv"
    df.to_csv(out_path, index=False)

    report_path = ANALYSIS_DIR / "cdp_content_analysis_summary.txt"
    with open(report_path, "w") as f:
        f.write("CDP CONTENT ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(df)}\n\n")
        f.write(f"Mean score 1 utterances: {df['n_cdp_score1'].mean():.2f}\n")
        f.write(f"Mean score 2 utterances: {df['n_cdp_score2'].mean():.2f}\n")
        f.write(f"Mean % score 1: {df['pct_score1'].mean():.1f}%\n")
        f.write(f"Mean % score 2: {df['pct_score2'].mean():.1f}%\n\n")
        f.write(f"Mean token count (score 1): {df['mean_tokens_score1'].mean():.2f}\n")
        f.write(f"Mean token count (score 2): {df['mean_tokens_score2'].mean():.2f}\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
