#!/usr/bin/env python
"""CDP by Cohort Analysis

Compares CDP entropy distributions across conference years (2020, 2021, 2022).
Runs ANOVA and Kruskal-Wallis to test for year effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def find_latest_entropy_csv() -> Optional[Path]:
    csvs = list(TABLES_DIR.glob("cdp_entropy_by_session_*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def extract_year_from_conference(conf: str) -> Optional[int]:
    """Extract year from conference code like '2020NES', '2021ABI'."""
    try:
        return int(conf[:4])
    except ValueError:
        return None


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return ranks with average for ties."""
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


def kruskal_wallis(*groups: np.ndarray) -> dict:
    """Kruskal-Wallis H test."""
    combined = np.concatenate(groups)
    ranks = rankdata(combined)

    n_total = len(combined)
    n_groups = len(groups)
    h = 0.0
    offset = 0

    for group in groups:
        n = len(group)
        r_sum = np.sum(ranks[offset : offset + n])
        h += r_sum**2 / n
        offset += n

    h = 12 / (n_total * (n_total + 1)) * h - 3 * (n_total + 1)
    return {"h": float(h), "n_groups": n_groups}


def main() -> None:
    csv_path = find_latest_entropy_csv()
    if csv_path is None or not csv_path.exists():
        print("ERROR: No entropy CSV found.")
        return

    df = pd.read_csv(csv_path)
    df["year"] = df["conference"].apply(extract_year_from_conference)
    df = df[df["year"].notna()]

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = ANALYSIS_DIR / "cdp_by_cohort_summary.txt"

    with open(report_path, "w") as f:
        f.write("CDP ENTROPY BY COHORT (YEAR)\n")
        f.write("=" * 80 + "\n\n")

        for metric in ["entropy_beginning", "entropy_middle", "entropy_end"]:
            if metric not in df.columns:
                continue

            f.write(f"{metric.upper()}\n")
            f.write("-" * 80 + "\n")

            for year in sorted(df["year"].unique()):
                subset = df[df["year"] == year][metric].dropna()
                if len(subset) > 0:
                    f.write(
                        f"  {int(year)}: n={len(subset)}, "
                        f"mean={subset.mean():.4f}, "
                        f"median={subset.median():.4f}, "
                        f"std={subset.std(ddof=1):.4f}\n"
                    )

            # Kruskal-Wallis test
            groups = [df[df["year"] == year][metric].dropna().values for year in sorted(df["year"].unique())]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                kw_result = kruskal_wallis(*groups)
                f.write(f"  Kruskal-Wallis H: {kw_result['h']:.4f}\n")
            f.write("\n")

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
