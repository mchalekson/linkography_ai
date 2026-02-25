#!/usr/bin/env python
"""Speaker Diversity vs Outcomes Analysis

Correlates Gini coefficients (speaker balance in CDP) with funding outcomes.
Tests hypothesis: Teams with balanced advanced coordination participation succeed more.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load speaker diversity metrics
    try:
        speaker_df = pd.read_csv(TABLES_DIR / "speaker_level_cdp.csv")
    except FileNotFoundError:
        print("ERROR: speaker_level_cdp.csv not found. Run 'make speaker_cdp' first.")
        return

    # Load outcomes
    try:
        outcomes_df = pd.read_csv(TABLES_DIR / "entropy_with_outcomes.csv")
    except FileNotFoundError:
        print("ERROR: entropy_with_outcomes.csv not found. Run 'make merge_outcomes' first.")
        return

    # Merge on session_id
    merged = speaker_df.merge(outcomes_df[["session_id", "any_funded", "funded_rate"]], on="session_id", how="inner")
    
    if len(merged) == 0:
        print("No matching sessions found.")
        return

    # Separate funded vs unfunded
    funded = merged[merged["any_funded"] == 1]
    unfunded = merged[merged["any_funded"] == 0]

    # Compute correlations
    corr_gini1_funded, p_corr_gini1 = spearmanr(merged["gini_score1"], merged["any_funded"])
    corr_gini2_funded, p_corr_gini2 = spearmanr(merged["gini_score2"], merged["any_funded"])
    corr_part_funded, p_corr_part = spearmanr(merged["speaker_participation_cdp"], merged["any_funded"])

    # Mann-Whitney U tests
    stat_gini1, p_gini1 = mannwhitneyu(funded["gini_score1"].dropna(), unfunded["gini_score1"].dropna())
    stat_gini2, p_gini2 = mannwhitneyu(funded["gini_score2"].dropna(), unfunded["gini_score2"].dropna())
    stat_part, p_part = mannwhitneyu(funded["speaker_participation_cdp"].dropna(), unfunded["speaker_participation_cdp"].dropna())

    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (x.mean() - y.mean()) / ((((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof) ** 0.5)

    d_gini1 = cohens_d(funded["gini_score1"].dropna(), unfunded["gini_score1"].dropna())
    d_gini2 = cohens_d(funded["gini_score2"].dropna(), unfunded["gini_score2"].dropna())
    d_part = cohens_d(funded["speaker_participation_cdp"].dropna(), unfunded["speaker_participation_cdp"].dropna())

    # Output summary
    report_path = ANALYSIS_DIR / "speaker_diversity_outcomes_summary.txt"
    with open(report_path, "w") as f:
        f.write("SPEAKER DIVERSITY vs OUTCOMES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(merged)}\n")
        f.write(f"Funded sessions: {len(funded)} ({100*len(funded)/len(merged):.1f}%)\n")
        f.write(f"Unfunded sessions: {len(unfunded)} ({100*len(unfunded)/len(merged):.1f}%)\n\n")

        f.write("GINI (Score 1 - Basic Coordination)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Funded mean: {funded['gini_score1'].mean():.3f} (std {funded['gini_score1'].std():.3f})\n")
        f.write(f"Unfunded mean: {unfunded['gini_score1'].mean():.3f} (std {unfunded['gini_score1'].std():.3f})\n")
        f.write(f"Spearman r: {corr_gini1_funded:.3f} (p={p_corr_gini1:.4f})\n")
        f.write(f"Mann-Whitney U: p={p_gini1:.4f}, Cohen's d={d_gini1:.3f}\n\n")

        f.write("GINI (Score 2 - Advanced Coordination)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Funded mean: {funded['gini_score2'].mean():.3f} (std {funded['gini_score2'].std():.3f})\n")
        f.write(f"Unfunded mean: {unfunded['gini_score2'].mean():.3f} (std {unfunded['gini_score2'].std():.3f})\n")
        f.write(f"Spearman r: {corr_gini2_funded:.3f} (p={p_corr_gini2:.4f})\n")
        f.write(f"Mann-Whitney U: p={p_gini2:.4f}, Cohen's d={d_gini2:.3f}\n\n")

        f.write("SPEAKER PARTICIPATION RATE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Funded mean: {funded['speaker_participation_cdp'].mean():.3f} (std {funded['speaker_participation_cdp'].std():.3f})\n")
        f.write(f"Unfunded mean: {unfunded['speaker_participation_cdp'].mean():.3f} (std {unfunded['speaker_participation_cdp'].std():.3f})\n")
        f.write(f"Spearman r: {corr_part_funded:.3f} (p={p_corr_part:.4f})\n")
        f.write(f"Mann-Whitney U: p={p_part:.4f}, Cohen's d={d_part:.3f}\n\n")

        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        if p_gini2 < 0.05:
            f.write("✓ SIGNIFICANT: Advanced coordination balance (Gini score 2) differs between funded/unfunded\n")
            if funded['gini_score2'].mean() < unfunded['gini_score2'].mean():
                f.write("  → Funded teams show LOWER Gini (more balanced): advanced coordination is more distributed\n")
            else:
                f.write("  → Funded teams show HIGHER Gini (more concentrated): advanced coordination is more centralized\n")
        else:
            f.write("✗ NOT SIGNIFICANT: Advanced coordination balance does not predict funding\n")

        if p_gini1 < 0.05:
            f.write("✓ SIGNIFICANT: Basic coordination balance (Gini score 1) differs between funded/unfunded\n")
        else:
            f.write("✗ NOT SIGNIFICANT: Basic coordination balance does not predict funding\n")

        if p_part < 0.05:
            f.write("✓ SIGNIFICANT: Overall speaker participation rate differs between funded/unfunded\n")
        else:
            f.write("✗ NOT SIGNIFICANT: Overall speaker participation rate does not predict funding\n")

    # Save merged data for further analysis
    merged_sorted = merged.sort_values("any_funded", ascending=False)
    merged_sorted.to_csv(TABLES_DIR / "speaker_diversity_with_outcomes.csv", index=False)

    print(f"Saved: {report_path}")
    print(f"Saved: {TABLES_DIR / 'speaker_diversity_with_outcomes.csv'}")


if __name__ == "__main__":
    main()
