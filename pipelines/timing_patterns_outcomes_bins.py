#!/usr/bin/env python
"""Timing patterns vs outcomes for multiple bin sizes."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def analyze_session_timing(session_bins: pd.DataFrame) -> dict | None:
    if len(session_bins) == 0:
        return None
    entropies = session_bins["entropy"].values
    n_bins = len(session_bins)

    pure_bins = (entropies < 0.2).sum()
    mixed_bins = (entropies >= 0.4).sum()
    purity_ratio = pure_bins / n_bins if n_bins > 0 else 0
    mixed_ratio = mixed_bins / n_bins if n_bins > 0 else 0

    entropy_diffs = abs(entropies[1:] - entropies[:-1])
    mean_jump = entropy_diffs.mean() if len(entropy_diffs) > 0 else 0
    max_jump = entropy_diffs.max() if len(entropy_diffs) > 0 else 0
    n_large_jumps = (entropy_diffs > 0.3).sum()

    if n_bins >= 3:
        first_third_entropy = entropies[: n_bins // 3].mean()
        last_third_entropy = entropies[2 * n_bins // 3 :].mean()
    else:
        first_third_entropy = entropies.mean() if n_bins > 0 else 0
        last_third_entropy = entropies.mean() if n_bins > 0 else 0

    entropy_trend = last_third_entropy - first_third_entropy

    return {
        "n_bins": n_bins,
        "mean_entropy": entropies.mean(),
        "purity_ratio": purity_ratio,
        "mixed_ratio": mixed_ratio,
        "mean_jump": mean_jump,
        "max_jump": max_jump,
        "n_transitions": n_large_jumps,
        "entropy_trend": entropy_trend,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-sec", type=int, required=True)
    args = parser.parse_args()

    timing_path = TABLES_DIR / f"cdp_fine_grained_entropy_{args.bin_sec}s.csv"
    if not timing_path.exists():
        print(f"Missing {timing_path}. Run fine_grained_cdp_timing.py --bin-sec {args.bin_sec}.")
        return

    timing_df = pd.read_csv(timing_path)
    outcomes_path = TABLES_DIR / "entropy_with_outcomes.csv"
    if not outcomes_path.exists():
        print("Missing entropy_with_outcomes.csv. Run merge_outcomes first.")
        return

    outcomes_df = pd.read_csv(outcomes_path)

    session_features = []
    for session_id, group in timing_df.groupby("session_id"):
        features = analyze_session_timing(group)
        if features is not None:
            features["session_id"] = session_id
            session_features.append(features)

    timing_features_df = pd.DataFrame(session_features)
    merged = timing_features_df.merge(
        outcomes_df[["session_id", "any_funded", "funded_rate"]],
        on="session_id",
        how="inner",
    )

    funded = merged[merged["any_funded"] == 1]
    unfunded = merged[merged["any_funded"] == 0]

    features_to_test = [
        "purity_ratio",
        "mixed_ratio",
        "mean_jump",
        "max_jump",
        "n_transitions",
        "entropy_trend",
    ]

    report_path = ANALYSIS_DIR / f"timing_patterns_outcomes_{args.bin_sec}s_summary.txt"
    with open(report_path, "w") as f:
        f.write("MEETING TIMING PATTERNS vs OUTCOMES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Bin size: {args.bin_sec} seconds\n")
        f.write(f"Sessions analyzed: {len(merged)}\n")
        f.write(f"Funded sessions: {len(funded)}\n")
        f.write(f"Unfunded sessions: {len(unfunded)}\n\n")

        for feature in features_to_test:
            f_vals = funded[feature].dropna()
            u_vals = unfunded[feature].dropna()
            if len(f_vals) == 0 or len(u_vals) == 0:
                continue
            stat, p_val = mannwhitneyu(f_vals, u_vals)
            f.write(f"{feature}\n")
            f.write(f"  Funded:   {f_vals.mean():.3f} ± {f_vals.std():.3f}\n")
            f.write(f"  Unfunded: {u_vals.mean():.3f} ± {u_vals.std():.3f}\n")
            f.write(f"  p-value: {p_val:.4f}\n\n")

    merged.to_csv(TABLES_DIR / f"timing_features_with_outcomes_{args.bin_sec}s.csv", index=False)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
