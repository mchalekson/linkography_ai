#!/usr/bin/env python
"""Fuzzy Linkography on data-v2 JSON annotations.

Builds weighted semantic links between utterances and exports chunk/session-level
fuzzy linkography metrics for downstream modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from linkography_ai.fuzzy_linkography import (
    compute_fuzzy_metrics,
    group_v2_chunk_files,
    load_chunk_moves,
)
from linkography_ai.paths import data_v2_root, display_path, outputs_root

DEFAULT_INPUT_ROOT = data_v2_root()
OUT_DIR = outputs_root()
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fuzzy linkography metrics for data-v2")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root folder for v2 chunk JSONs")
    parser.add_argument("--conference", type=str, default="ALL", help="Conference code or ALL")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold for fuzzy links")
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="lsa",
        choices=["tfidf", "lsa", "hybrid"],
        help="Semantic similarity method for fuzzy links",
    )
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    grouped = group_v2_chunk_files(args.input_root)
    if not grouped:
        print(f"No chunk files found under: {args.input_root}")
        return

    chunk_rows: List[Dict[str, object]] = []
    meeting_rows: List[Dict[str, object]] = []

    for group in grouped.values():
        conference = str(group["conference"])
        if args.conference.upper() != "ALL" and conference != args.conference:
            continue

        session_id = str(group["session_id"])
        meeting_id = str(group["meeting_id"])
        chunk_files = group["chunk_files"]

        all_moves = []
        for chunk_index, chunk_fp in chunk_files:
            chunk_moves = load_chunk_moves(chunk_fp, chunk_index=chunk_index)
            all_moves.extend(chunk_moves)

            metrics = compute_fuzzy_metrics(
                chunk_moves,
                threshold=args.threshold,
                similarity_method=args.similarity_method,
            )
            row = {
                "conference": conference,
                "session_id": session_id,
                "meeting_id": meeting_id,
                "chunk_index": int(chunk_index),
                "chunk_file": display_path(chunk_fp, base=args.input_root),
            }
            row.update(metrics)
            chunk_rows.append(row)

        all_moves.sort(key=lambda m: (m.start_sec, m.chunk_index, m.utterance_index))
        meeting_metrics = compute_fuzzy_metrics(
            all_moves,
            threshold=args.threshold,
            similarity_method=args.similarity_method,
        )
        meeting_row = {
            "conference": conference,
            "session_id": session_id,
            "meeting_id": meeting_id,
            "n_chunks": len(chunk_files),
        }
        meeting_row.update(meeting_metrics)
        meeting_rows.append(meeting_row)

    if not meeting_rows:
        print("No meetings matched requested conference filter.")
        return

    chunk_df = pd.DataFrame(chunk_rows).sort_values(["conference", "session_id", "meeting_id", "chunk_index"])
    meeting_df = pd.DataFrame(meeting_rows).sort_values(["conference", "session_id", "meeting_id"])

    chunk_out = TABLES_DIR / "fuzzy_linkography_v2_by_chunk.csv"
    meeting_out = TABLES_DIR / "fuzzy_linkography_v2_by_meeting.csv"
    chunk_df.to_csv(chunk_out, index=False)
    meeting_df.to_csv(meeting_out, index=False)

    report_out = ANALYSIS_DIR / "fuzzy_linkography_v2_summary.txt"
    with open(report_out, "w") as f:
        f.write("FUZZY LINKOGRAPHY (V2 JSON) SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Input root: {display_path(args.input_root)}\n")
        f.write(f"Conference filter: {args.conference}\n")
        f.write(f"Similarity threshold: {args.threshold:.3f}\n\n")
        f.write(f"Similarity method: {args.similarity_method}\n\n")
        f.write(f"Meetings analyzed: {len(meeting_df)}\n")
        f.write(f"Chunks analyzed: {len(chunk_df)}\n\n")

        keys = [
            "n_moves",
            "weighted_ldi",
            "mean_nonzero_weight",
            "cross_speaker_weight_ratio",
            "late_minus_early_backlink",
            "overall_link_entropy",
        ]
        f.write("Meeting-level metric means:\n")
        for k in keys:
            if k in meeting_df.columns:
                f.write(f"  - {k}: {meeting_df[k].mean():.4f}\n")

    print(f"Saved: {chunk_out}")
    print(f"Saved: {meeting_out}")
    print(f"Saved: {report_out}")


if __name__ == "__main__":
    main()
