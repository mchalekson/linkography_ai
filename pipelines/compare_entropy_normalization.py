#!/usr/bin/env python
"""Compare normalized vs raw entropy.

Uses the latest entropy CSV and reconstructs raw entropy from normalized values
using K = n_unique_cdp_*.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"
FIGURES_DIR = REPO_ROOT / "figures" / "final"


def find_latest_entropy_csv() -> Optional[Path]:
    csvs = list(TABLES_DIR.glob("cdp_entropy_by_session_*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def reconstruct_raw(normalized: pd.Series, k: pd.Series) -> pd.Series:
    raw = []
    for n, kk in zip(normalized, k):
        if pd.isna(n) or pd.isna(kk):
            raw.append(np.nan)
            continue
        if kk <= 1:
            raw.append(0.0)
            continue
        raw.append(float(n) * math.log(kk, 2))
    return pd.Series(raw, index=normalized.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare normalized vs raw entropy")
    parser.add_argument("--csv", type=Path, help="Path to entropy CSV")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_entropy_csv()
    if csv_path is None or not csv_path.exists():
        print("ERROR: No entropy CSV found.")
        return

    df = pd.read_csv(csv_path)

    rows = []
    for seg in ["beginning", "middle", "end"]:
        norm_col = f"entropy_{seg}"
        k_col = f"n_unique_cdp_{seg}"
        if norm_col not in df.columns or k_col not in df.columns:
            continue
        raw = reconstruct_raw(df[norm_col], df[k_col])
        diff = raw - df[norm_col]
        rows.append({
            "segment": seg,
            "mean_diff": float(diff.mean()),
            "median_diff": float(diff.median()),
            "std_diff": float(diff.std(ddof=1)),
        })

        df[f"entropy_raw_{seg}"] = raw

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    report_path = ANALYSIS_DIR / "entropy_normalization_comparison.txt"
    with open(report_path, "w") as f:
        f.write("ENTROPY NORMALIZATION COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source CSV: {csv_path}\n\n")
        for r in rows:
            f.write(f"{r['segment'].upper()}\n")
            f.write(f"  Mean diff (raw - normalized): {r['mean_diff']:.6f}\n")
            f.write(f"  Median diff: {r['median_diff']:.6f}\n")
            f.write(f"  Std diff: {r['std_diff']:.6f}\n\n")

    # Scatter plot (beginning only, representative)
    if "entropy_raw_beginning" in df.columns:
        fig_path = FIGURES_DIR / "raw_vs_normalized_entropy_scatter.png"
        plt.figure(figsize=(5, 4))
        plt.scatter(df["entropy_beginning"], df["entropy_raw_beginning"], alpha=0.5)
        plt.xlabel("Normalized entropy (beginning)")
        plt.ylabel("Raw entropy (beginning)")
        plt.title("Raw vs Normalized Entropy")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    else:
        fig_path = None

    print(f"Saved: {report_path}")
    if fig_path:
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
