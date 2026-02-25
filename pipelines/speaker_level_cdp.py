#!/usr/bin/env python
"""Speaker-Level CDP Analysis

Extracts speaker identities and analyzes their CDP score distributions.
Computes diversity metrics (Gini coefficient) for each session.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from linkography_ai.discovery import list_conferences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient (0=equal, 1=concentrated)."""
    if not values or sum(values) == 0:
        return math.nan
    n = len(values)
    sorted_v = sorted(values)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * cumsum) / (n * sum(values)) - (n + 1) / n


def extract_speaker_cdp(session_json: dict) -> dict:
    """Map speaker -> CDP score counts."""
    speaker_cdp = {}

    for u in session_json.get("all_data", []):
        if not isinstance(u, dict):
            continue
        speaker = (u.get("speaker") or "").strip()
        if not speaker:
            continue

        ann = u.get("annotations") or u.get("annotation_dict") or {}
        if not isinstance(ann, dict):
            continue

        cdp_data = ann.get("Coordination and Decision Practices")
        if not isinstance(cdp_data, dict):
            continue

        score = cdp_data.get("score")
        if score not in (1, 2):
            continue

        if speaker not in speaker_cdp:
            speaker_cdp[speaker] = {"score1": 0, "score2": 0}
        if score == 1:
            speaker_cdp[speaker]["score1"] += 1
        else:
            speaker_cdp[speaker]["score2"] += 1

    return speaker_cdp


def analyze_session(session_path: Path) -> Optional[dict]:
    try:
        obj = json.loads(session_path.read_text())
    except Exception:
        return None

    session_json = obj if isinstance(obj, dict) and "all_data" in obj else None
    if session_json is None:
        return None

    speaker_cdp = extract_speaker_cdp(session_json)
    if not speaker_cdp:
        return None

    n_speakers_with_cdp = len(speaker_cdp)
    total_score1 = sum(s["score1"] for s in speaker_cdp.values())
    total_score2 = sum(s["score2"] for s in speaker_cdp.values())

    # Compute Gini for score 1 usage across speakers
    score1_counts = [s["score1"] for s in speaker_cdp.values() if s["score1"] > 0]
    gini_score1 = gini_coefficient(score1_counts) if score1_counts else math.nan

    score2_counts = [s["score2"] for s in speaker_cdp.values() if s["score2"] > 0]
    gini_score2 = gini_coefficient(score2_counts) if score2_counts else math.nan

    # Speaker balance: how many speakers contribute to CDP?
    all_speakers = set(u.get("speaker", "").strip() for u in session_json.get("all_data", [])
                      if isinstance(u, dict) and u.get("speaker"))
    speaker_participation = len(speaker_cdp) / len(all_speakers) if all_speakers else math.nan

    return {
        "session_id": session_path.stem,
        "n_speakers_with_cdp": n_speakers_with_cdp,
        "n_speakers_total": len(all_speakers),
        "speaker_participation_cdp": speaker_participation,
        "total_cdp_score1": total_score1,
        "total_cdp_score2": total_score2,
        "gini_score1": gini_score1,
        "gini_score2": gini_score2,
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences()
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
    out_path = TABLES_DIR / "speaker_level_cdp.csv"
    df.to_csv(out_path, index=False)

    report_path = ANALYSIS_DIR / "speaker_level_cdp_summary.txt"
    with open(report_path, "w") as f:
        f.write("SPEAKER-LEVEL CDP ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(df)}\n\n")
        f.write(f"Mean speakers with CDP: {df['n_speakers_with_cdp'].mean():.2f}\n")
        f.write(f"Mean total speakers: {df['n_speakers_total'].mean():.2f}\n")
        f.write(f"Mean CDP participation: {df['speaker_participation_cdp'].mean():.2f}\n\n")
        f.write(f"Gini (score 1 concentration): {df['gini_score1'].mean():.3f}\n")
        f.write(f"Gini (score 2 concentration): {df['gini_score2'].mean():.3f}\n")
        f.write("(Lower Gini = more balanced; higher = concentrated)\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
