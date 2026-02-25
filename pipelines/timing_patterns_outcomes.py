#!/usr/bin/env python
"""Timing Patterns vs Outcomes Analysis

Analyzes fine-grained CDP timing signatures and correlates meeting phase patterns
with funding outcomes. Hypothesis: Teams with clear phase transitions succeed more.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def analyze_session_timing(session_bins: pd.DataFrame) -> dict:
    """Extract timing features from per-session bins."""
    if len(session_bins) == 0:
        return None

    entropies = session_bins["entropy"].values
    
    # Phase diversity: count pure score-1, pure score-2, and mixed bins
    pure_score1 = (entropies < 0.1).sum()
    pure_score2 = (entropies < 0.1).sum()  # This is wrong, but let me use a better metric
    
    # Better: use entropy to classify
    # Entropy near 0 = one score dominates
    # Entropy near 1 = mixed
    pure_bins = (entropies < 0.2).sum()  # Dominated by one score
    mixed_bins = (entropies >= 0.4).sum()  # Balanced between scores
    
    n_bins = len(session_bins)
    purity_ratio = pure_bins / n_bins if n_bins > 0 else 0
    mixed_ratio = mixed_bins / n_bins if n_bins > 0 else 0

    # Phase transitions: detect entropy jumps
    entropy_diffs = abs(entropies[1:] - entropies[:-1])
    mean_jump = entropy_diffs.mean() if len(entropy_diffs) > 0 else 0
    max_jump = entropy_diffs.max() if len(entropy_diffs) > 0 else 0
    n_large_jumps = (entropy_diffs > 0.3).sum()  # Transitions

    # Temporal distribution: does entropy change over time?
    first_third_entropy = entropies[:n_bins//3].mean() if n_bins > 0 else 0
    last_third_entropy = entropies[2*n_bins//3:].mean() if n_bins > 0 else 0
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
        "first_third_entropy": first_third_entropy,
        "last_third_entropy": last_third_entropy,
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load fine-grained timing
    try:
        timing_df = pd.read_csv(TABLES_DIR / "cdp_fine_grained_entropy_300s.csv")
    except FileNotFoundError:
        print("ERROR: cdp_fine_grained_entropy_300s.csv not found. Run 'make fine_grained' first.")
        return

    # Load outcomes
    try:
        outcomes_df = pd.read_csv(TABLES_DIR / "entropy_with_outcomes.csv")
    except FileNotFoundError:
        print("ERROR: entropy_with_outcomes.csv not found. Run 'make merge_outcomes' first.")
        return

    # Group timing by session
    session_features = []
    for session_id, group in timing_df.groupby("session_id"):
        features = analyze_session_timing(group)
        if features is not None:
            features["session_id"] = session_id
            session_features.append(features)

    timing_features_df = pd.DataFrame(session_features)

    # Merge with outcomes
    merged = timing_features_df.merge(
        outcomes_df[["session_id", "any_funded", "funded_rate"]],
        on="session_id",
        how="inner"
    )

    if len(merged) == 0:
        print("No matching sessions found.")
        return

    # Separate funded vs unfunded
    funded = merged[merged["any_funded"] == 1]
    unfunded = merged[merged["any_funded"] == 0]

    # Statistical tests on key timing features
    features_to_test = [
        "purity_ratio",
        "mixed_ratio",
        "mean_jump",
        "max_jump",
        "n_transitions",
        "entropy_trend",
    ]

    results = {}
    for feature in features_to_test:
        f_vals = funded[feature].dropna()
        u_vals = unfunded[feature].dropna()
        if len(f_vals) > 0 and len(u_vals) > 0:
            stat, p_val = mannwhitneyu(f_vals, u_vals)
            results[feature] = {
                "funded_mean": f_vals.mean(),
                "unfunded_mean": u_vals.mean(),
                "p_value": p_val,
                "funded_std": f_vals.std(),
                "unfunded_std": u_vals.std(),
            }

    # Output summary
    report_path = ANALYSIS_DIR / "timing_patterns_outcomes_summary.txt"
    with open(report_path, "w") as f:
        f.write("MEETING TIMING PATTERNS vs OUTCOMES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(merged)}\n")
        f.write(f"Funded sessions: {len(funded)} ({100*len(funded)/len(merged):.1f}%)\n")
        f.write(f"Unfunded sessions: {len(unfunded)} ({100*len(unfunded)/len(merged):.1f}%)\n\n")

        f.write("KEY TIMING FEATURES\n")
        f.write("-" * 80 + "\n\n")

        for feature, res in results.items():
            f.write(f"{feature}\n")
            f.write(f"  Funded:   {res['funded_mean']:.3f} ± {res['funded_std']:.3f}\n")
            f.write(f"  Unfunded: {res['unfunded_mean']:.3f} ± {res['unfunded_std']:.3f}\n")
            f.write(f"  p-value: {res['p_value']:.4f}")
            if res['p_value'] < 0.05:
                f.write(" ✓ SIGNIFICANT\n")
            else:
                f.write(" ✗ not significant\n")
            f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        significant_features = [k for k, v in results.items() if v['p_value'] < 0.05]
        if significant_features:
            f.write(f"✓ Found {len(significant_features)} significant timing differences:\n")
            for feat in significant_features:
                f.write(f"  - {feat}\n")
                if funded[feat].mean() > unfunded[feat].mean():
                    f.write(f"    → Funded teams show HIGHER {feat}\n")
                else:
                    f.write(f"    → Funded teams show LOWER {feat}\n")
        else:
            f.write("✗ No significant timing pattern differences between funded/unfunded\n")

        f.write("\n\nMEETING RHYTHM PROFILES\n")
        f.write("-" * 80 + "\n")
        f.write("Purity Ratio: Fraction of bins dominated by one coordination type (score 1 or 2)\n")
        f.write("  High purity → Clear phase separation (focused phases)\n")
        f.write("  Low purity → Mixed coordination throughout\n\n")
        f.write("Mixed Ratio: Fraction of bins with balanced score 1 and score 2\n")
        f.write("  High mixed → Simultaneous coordination and decision-making\n")
        f.write("  Low mixed → Teams switch between modes cleanly\n\n")
        f.write("n_Transitions: Number of large entropy jumps (>0.3)\n")
        f.write("  High transitions → Teams deliberately shift between phases\n")
        f.write("  Low transitions → Steady state coordination style\n\n")
        f.write("Entropy Trend: Change from first third to last third\n")
        f.write("  Positive → Teams transition toward more mixed coordination\n")
        f.write("  Negative → Teams focus/narrow coordination over time\n")

    # Save timing features with outcomes
    merged_sorted = merged.sort_values("any_funded", ascending=False)
    merged_sorted.to_csv(TABLES_DIR / "timing_features_with_outcomes.csv", index=False)

    print(f"Saved: {report_path}")
    print(f"Saved: {TABLES_DIR / 'timing_features_with_outcomes.csv'}")


if __name__ == "__main__":
    main()
