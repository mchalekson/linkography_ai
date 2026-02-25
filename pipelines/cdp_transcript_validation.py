#!/usr/bin/env python
"""Qualitative transcript validation for CDP findings.

Samples transcript excerpts to validate quantitative CDP findings (entropy + Gini).
Outputs a CSV of excerpts and a readable TXT summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def load_session_json(conference: str, session_id: str) -> dict | None:
    session_path = DATA_DIR / conference / "session_data" / f"{session_id}.json"
    if not session_path.exists():
        return None
    try:
        return json.loads(session_path.read_text())
    except Exception:
        return None


def iter_utterances(session_json: dict) -> List[dict]:
    if isinstance(session_json, dict) and "all_data" in session_json:
        data = session_json.get("all_data", [])
    elif isinstance(session_json, list):
        data = session_json
    else:
        data = []
    return [u for u in data if isinstance(u, dict)]


def extract_cdp_utterances(session_json: dict) -> List[dict]:
    utterances = []
    for u in iter_utterances(session_json):
        ann = u.get("annotations") or u.get("annotation_dict") or {}
        if not isinstance(ann, dict):
            continue
        cdp = ann.get("Coordination and Decision Practices")
        if not isinstance(cdp, dict):
            continue
        score = cdp.get("score")
        if score not in (1, 2):
            continue
        text = (u.get("transcript") or u.get("text") or u.get("utterance") or "").strip()
        if not text:
            continue
        speaker = (u.get("speaker") or "").strip()
        tokens = len(text.split())
        utterances.append(
            {
                "speaker": speaker,
                "score": int(score),
                "text": text,
                "tokens": tokens,
            }
        )
    return utterances


def sample_excerpts(utterances: List[dict], score: int, n: int = 3) -> List[dict]:
    score_utts = [u for u in utterances if u["score"] == score]
    # Prefer longer utterances for score 2 and shorter for score 1
    if score == 2:
        score_utts.sort(key=lambda x: x["tokens"], reverse=True)
    else:
        score_utts.sort(key=lambda x: x["tokens"])
    return score_utts[:n]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    speaker_path = TABLES_DIR / "speaker_level_cdp.csv"
    entropy_path = sorted(TABLES_DIR.glob("cdp_entropy_by_session_ALL_*.csv"))[-1]

    speaker_df = pd.read_csv(speaker_path)
    entropy_df = pd.read_csv(entropy_path)

    # Sample sessions by high/low Gini for score 2
    gini_sorted = speaker_df.dropna(subset=["gini_score2"]).sort_values("gini_score2")
    low_gini = gini_sorted.head(3)
    high_gini = gini_sorted.tail(3)

    # Sample 2022 sessions with lowest middle entropy
    entropy_df["year"] = entropy_df["conference"].astype(str).str.slice(0, 4)
    entropy_2022 = entropy_df[entropy_df["year"] == "2022"].dropna(subset=["entropy_middle"])
    low_mid_entropy_2022 = entropy_2022.sort_values("entropy_middle").head(3)

    samples = []

    def add_samples(label: str, rows: pd.DataFrame) -> None:
        for _, row in rows.iterrows():
            conference = row["conference"]
            session_id = row["session_id"]
            session_json = load_session_json(conference, session_id)
            if not session_json:
                continue
            utterances = extract_cdp_utterances(session_json)
            if not utterances:
                continue
            s1 = sample_excerpts(utterances, score=1, n=3)
            s2 = sample_excerpts(utterances, score=2, n=3)
            for u in s1 + s2:
                samples.append(
                    {
                        "label": label,
                        "conference": conference,
                        "session_id": session_id,
                        "speaker": u["speaker"],
                        "score": u["score"],
                        "tokens": u["tokens"],
                        "text": u["text"],
                    }
                )

    add_samples("low_gini_score2", low_gini)
    add_samples("high_gini_score2", high_gini)
    add_samples("2022_low_mid_entropy", low_mid_entropy_2022)

    samples_df = pd.DataFrame(samples)
    samples_csv = TABLES_DIR / "cdp_transcript_validation_samples.csv"
    samples_df.to_csv(samples_csv, index=False)

    # Write readable summary
    summary_path = ANALYSIS_DIR / "cdp_transcript_validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("CDP TRANSCRIPT VALIDATION SAMPLES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source speaker metrics: {speaker_path}\n")
        f.write(f"Source entropy: {entropy_path}\n\n")

        for label in ["low_gini_score2", "high_gini_score2", "2022_low_mid_entropy"]:
            subset = samples_df[samples_df["label"] == label]
            f.write(f"\n[{label}]\n")
            f.write("-" * 80 + "\n")
            for session_id in subset["session_id"].unique():
                f.write(f"Session: {session_id}\n")
                ses = subset[subset["session_id"] == session_id]
                for _, row in ses.iterrows():
                    f.write(
                        f"  (score {row['score']}, {row['tokens']} tokens) {row['speaker']}: {row['text']}\n"
                    )
                f.write("\n")

    print(f"Saved: {samples_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
