#!/usr/bin/env python3
"""Inspect representative meetings with high vs low mean_nonzero_weight."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def main() -> None:
    in_path = TABLES_DIR / "fuzzy_linkography_with_outcomes_by_meeting.csv"
    df = pd.read_csv(in_path)
    if df.empty:
        print("No meeting-level fuzzy table found.")
        return

    df["session_id_norm"] = df["session_id"].astype(str).str.replace("^output_", "", regex=True)
    df = df[df["n_moves"].fillna(0) >= 10].copy()
    ranked = df.sort_values("mean_nonzero_weight", ascending=False).reset_index(drop=True)

    high = ranked.head(10).copy()
    low = ranked.tail(10).sort_values("mean_nonzero_weight", ascending=True).copy()

    compare = pd.concat(
        [
            high.assign(example_group="high"),
            low.assign(example_group="low"),
        ],
        ignore_index=True,
    )
    cols = [
        "example_group",
        "conference",
        "session_id_norm",
        "meeting_id",
        "n_chunks",
        "n_moves",
        "mean_nonzero_weight",
        "weighted_ldi",
        "cross_speaker_weight_ratio",
        "overall_link_entropy",
        "funded_rate",
        "any_funded",
    ]
    compare = compare[[c for c in cols if c in compare.columns]]

    out_csv = TABLES_DIR / "fuzzy_linkography_example_meetings.csv"
    out_txt = ANALYSIS_DIR / "fuzzy_linkography_example_meetings.txt"
    compare.to_csv(out_csv, index=False)

    with open(out_txt, "w") as f:
        f.write("FUZZY LINKOGRAPHY EXAMPLE MEETINGS\n")
        f.write("=" * 72 + "\n\n")
        f.write("Highest mean_nonzero_weight meetings:\n\n")
        for _, row in high.iterrows():
            f.write(f"{row['meeting_id']}\n")
            f.write(f"  conference={row['conference']} session={row['session_id_norm']}\n")
            f.write(f"  n_moves={row['n_moves']:.0f} n_chunks={row['n_chunks']:.0f}\n")
            f.write(f"  mean_nonzero_weight={row['mean_nonzero_weight']:.4f}\n")
            f.write(f"  weighted_ldi={row['weighted_ldi']:.4f} cross_speaker_ratio={row['cross_speaker_weight_ratio']:.4f}\n")
            if pd.notna(row.get("any_funded")):
                f.write(f"  any_funded={int(row['any_funded'])} funded_rate={row['funded_rate']:.3f}\n")
            f.write("\n")

        f.write("Lowest mean_nonzero_weight meetings:\n\n")
        for _, row in low.iterrows():
            f.write(f"{row['meeting_id']}\n")
            f.write(f"  conference={row['conference']} session={row['session_id_norm']}\n")
            f.write(f"  n_moves={row['n_moves']:.0f} n_chunks={row['n_chunks']:.0f}\n")
            f.write(f"  mean_nonzero_weight={row['mean_nonzero_weight']:.4f}\n")
            f.write(f"  weighted_ldi={row['weighted_ldi']:.4f} cross_speaker_ratio={row['cross_speaker_weight_ratio']:.4f}\n")
            if pd.notna(row.get("any_funded")):
                f.write(f"  any_funded={int(row['any_funded'])} funded_rate={row['funded_rate']:.3f}\n")
            f.write("\n")

        f.write("Suggested manual follow-up:\n")
        f.write("  Open the corresponding data-v2 session folders and compare the high vs low meetings qualitatively.\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
