#!/usr/bin/env python
"""Merge fuzzy linkography outputs with outcome labels.

Inputs:
  - outputs/tables/fuzzy_linkography_v2_by_meeting.csv
  - outputs/tables/entropy_with_outcomes.csv

Outputs:
  - outputs/tables/fuzzy_linkography_with_outcomes_by_meeting.csv
  - outputs/tables/fuzzy_linkography_with_outcomes_by_session.csv
  - outputs/analysis/fuzzy_linkography_with_outcomes_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def _normalize_session_id(s: object) -> str:
    txt = str(s or "")
    return txt[len("output_") :] if txt.startswith("output_") else txt


def main() -> None:
    fuzzy_path = TABLES_DIR / "fuzzy_linkography_v2_by_meeting.csv"
    outcomes_path = TABLES_DIR / "entropy_with_outcomes.csv"

    if not fuzzy_path.exists():
        print(f"ERROR: Missing fuzzy table: {fuzzy_path}")
        print("Run: make fuzzy_v2")
        return
    if not outcomes_path.exists():
        print(f"ERROR: Missing outcomes table: {outcomes_path}")
        print("Run: make merge_outcomes")
        return

    fuzzy = pd.read_csv(fuzzy_path)
    outcomes = pd.read_csv(outcomes_path)

    fuzzy["session_id_norm"] = fuzzy["session_id"].map(_normalize_session_id)
    outcomes["session_id_norm"] = outcomes["session_id"].astype(str)

    outcome_cols = [
        "conference",
        "session_id_norm",
        "funded_rate",
        "any_funded",
        "n_teams",
    ]
    outcome_small = outcomes[outcome_cols].drop_duplicates(subset=["conference", "session_id_norm"])

    merged_meeting = fuzzy.merge(
        outcome_small,
        on=["conference", "session_id_norm"],
        how="left",
    )

    feature_cols = [
        "n_moves",
        "weighted_ldi",
        "mean_nonzero_weight",
        "forelink_weight_mean",
        "backlink_weight_mean",
        "cross_speaker_weight_ratio",
        "late_minus_early_backlink",
        "forelink_entropy",
        "backlink_entropy",
        "horizon_entropy",
        "overall_link_entropy",
    ]

    def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
        m = series.notna() & weights.notna()
        if not m.any():
            return np.nan
        w = weights[m].astype(float)
        x = series[m].astype(float)
        if float(w.sum()) <= 0:
            return float(x.mean())
        return float(np.average(x, weights=w))

    session_rows = []
    group_cols = ["conference", "session_id_norm"]
    for (conf, sid), g in merged_meeting.groupby(group_cols, dropna=False):
        row = {
            "conference": conf,
            "session_id": sid,
            "n_meetings": int(len(g)),
            "n_chunks_sum": float(g["n_chunks"].fillna(0).sum()),
            "n_moves_sum": float(g["n_moves"].fillna(0).sum()),
            "funded_rate": g["funded_rate"].dropna().iloc[0] if g["funded_rate"].notna().any() else np.nan,
            "any_funded": g["any_funded"].dropna().iloc[0] if g["any_funded"].notna().any() else np.nan,
            "n_teams": g["n_teams"].dropna().iloc[0] if g["n_teams"].notna().any() else np.nan,
        }
        for col in feature_cols:
            row[col] = _weighted_mean(g[col], g["n_moves"])
        session_rows.append(row)

    merged_session = pd.DataFrame(session_rows).sort_values(["conference", "session_id"]).reset_index(drop=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out_meeting = TABLES_DIR / "fuzzy_linkography_with_outcomes_by_meeting.csv"
    out_session = TABLES_DIR / "fuzzy_linkography_with_outcomes_by_session.csv"
    merged_meeting.to_csv(out_meeting, index=False)
    merged_session.to_csv(out_session, index=False)

    n_meeting_total = len(merged_meeting)
    n_meeting_matched = int(merged_meeting["any_funded"].notna().sum())
    n_session_total = len(merged_session)
    n_session_matched = int(merged_session["any_funded"].notna().sum())

    report = ANALYSIS_DIR / "fuzzy_linkography_with_outcomes_summary.txt"
    with open(report, "w") as f:
        f.write("FUZZY LINKOGRAPHY + OUTCOMES MERGE SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Meeting rows: {n_meeting_total}\n")
        f.write(f"Meeting rows matched to outcomes: {n_meeting_matched} ({100*n_meeting_matched/max(1,n_meeting_total):.1f}%)\n\n")
        f.write(f"Session rows: {n_session_total}\n")
        f.write(f"Session rows matched to outcomes: {n_session_matched} ({100*n_session_matched/max(1,n_session_total):.1f}%)\n\n")
        f.write("Primary fuzzy features available:\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")

    print(f"Saved: {out_meeting}")
    print(f"Saved: {out_session}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
