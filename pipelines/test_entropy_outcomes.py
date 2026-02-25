#!/usr/bin/env python
"""Entropy vs Outcomes Statistical Tests

Runs statistical comparisons between funded vs unfunded sessions using
entropy_with_outcomes.csv. Computes Mann-Whitney U, Cohen's d, and
correlation between funded_rate and entropy_end.

Usage:
    python pipelines/test_entropy_outcomes.py
    python pipelines/test_entropy_outcomes.py --csv outputs/tables/entropy_with_outcomes.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def find_entropy_with_outcomes_csv() -> Optional[Path]:
    """Find entropy_with_outcomes.csv in outputs/tables/."""
    path = TABLES_DIR / "entropy_with_outcomes.csv"
    return path if path.exists() else None


def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return ranks with average for ties (1-based)."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Mann-Whitney U and normal-approx p-value (two-sided)."""
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)
    r1 = np.sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Tie correction
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    n = n1 + n2
    var_u = n1 * n2 / 12 * ((n + 1) - tie_sum / (n * (n - 1))) if n > 1 else 0

    if var_u <= 0:
        z = float("nan")
        p = float("nan")
    else:
        z = (u - (n1 * n2 / 2)) / math.sqrt(var_u)
        p = 2 * (1 - normal_cdf(abs(z)))

    return {"u": float(u), "u1": float(u1), "u2": float(u2), "z": float(z), "p": float(p)}


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.std(x, ddof=1)
    s2 = np.std(y, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> dict:
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        r = np.corrcoef(x[idx], y[idx])[0, 1]
        boot.append(r)
    boot = np.array(boot)
    alpha = 1 - ci
    return {
        "mean": float(np.mean(boot)),
        "lower": float(np.percentile(boot, 100 * alpha / 2)),
        "upper": float(np.percentile(boot, 100 * (1 - alpha / 2))),
    }


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test entropy vs outcomes")
    parser.add_argument("--csv", type=Path, help="Path to entropy_with_outcomes.csv")
    args = parser.parse_args()

    csv_path = args.csv or find_entropy_with_outcomes_csv()
    if csv_path is None or not csv_path.exists():
        print("ERROR: entropy_with_outcomes.csv not found.")
        print("Run this first:")
        print("  python pipelines/merge_entropy_with_outcomes.py")
        return

    print(f"Loading merged data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter to sessions with outcome data and entropy_end present
    df = df[df["funded_rate"].notna() & df["entropy_end"].notna()]

    funded = df[df["any_funded"] == 1]["entropy_end"].astype(float).values
    unfunded = df[df["any_funded"] == 0]["entropy_end"].astype(float).values

    if len(funded) == 0 or len(unfunded) == 0:
        print("ERROR: Not enough data in funded/unfunded groups.")
        return

    # Group summary
    group_summary = pd.DataFrame([
        {
            "group": "any_funded=1",
            "n": len(funded),
            "mean_entropy_end": np.mean(funded),
            "median_entropy_end": np.median(funded),
            "std_entropy_end": np.std(funded, ddof=1),
        },
        {
            "group": "any_funded=0",
            "n": len(unfunded),
            "mean_entropy_end": np.mean(unfunded),
            "median_entropy_end": np.median(unfunded),
            "std_entropy_end": np.std(unfunded, ddof=1),
        },
    ])

    mw = mann_whitney_u(funded, unfunded)
    d = cohen_d(funded, unfunded)
    mean_diff = float(np.mean(funded) - np.mean(unfunded))

    # Correlations
    x = df["funded_rate"].astype(float).values
    y = df["entropy_end"].astype(float).values
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    pearson_ci = bootstrap_corr(x, y)
    spearman_r = spearman_corr(x, y)
    spearman_ci = bootstrap_corr(rankdata(x), rankdata(y))

    # Write outputs
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "entropy_outcome_group_summary.csv"
    group_summary.to_csv(summary_path, index=False)

    report_path = ANALYSIS_DIR / "entropy_outcomes_stats.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ENTROPY vs OUTCOMES STATISTICAL TESTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions with outcomes: {len(df)}\n")
        f.write(f"Funded (any_funded=1): {len(funded)}\n")
        f.write(f"Unfunded (any_funded=0): {len(unfunded)}\n\n")

        f.write("GROUP SUMMARY (entropy_end)\n")
        f.write("-" * 80 + "\n")
        for _, row in group_summary.iterrows():
            f.write(
                f"{row['group']}: n={int(row['n'])}, mean={row['mean_entropy_end']:.4f}, "
                f"median={row['median_entropy_end']:.4f}, std={row['std_entropy_end']:.4f}\n"
            )
        f.write("\n")

        f.write("MANN-WHITNEY U TEST (entropy_end)\n")
        f.write("-" * 80 + "\n")
        f.write(f"U: {mw['u']:.2f} (U1={mw['u1']:.2f}, U2={mw['u2']:.2f})\n")
        f.write(f"z: {mw['z']:.4f}\n")
        f.write(f"p (two-sided, normal approx): {mw['p']:.4f}\n")
        f.write(f"Mean difference (funded - unfunded): {mean_diff:.4f}\n")
        f.write(f"Cohen's d: {d:.4f}\n\n")

        f.write("CORRELATION (funded_rate vs entropy_end)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Pearson r: {pearson_r:.4f}\n")
        f.write(f"Pearson r (bootstrap 95% CI): [{pearson_ci['lower']:.4f}, {pearson_ci['upper']:.4f}]\n")
        f.write(f"Spearman rho: {spearman_r:.4f}\n")
        f.write(f"Spearman rho (bootstrap 95% CI): [{spearman_ci['lower']:.4f}, {spearman_ci['upper']:.4f}]\n")

    print(f"Saved group summary: {summary_path}")
    print(f"Saved stats report: {report_path}")


if __name__ == "__main__":
    main()
