#!/usr/bin/env python
"""Pairwise cohort tests for CDP entropy by meeting thirds.

Computes Mann-Whitney U tests (two-sided) for each pair of years
(2020 vs 2021 vs 2022) and applies Holm correction within each segment.
"""

from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(TABLES_DIR.glob("cdp_entropy_by_session_ALL_*.csv"))
    if not files:
        raise SystemExit("No cdp_entropy_by_session_ALL_*.csv found in outputs/tables")

    path = files[-1]
    df = pd.read_csv(path)
    df = df.copy()
    df["year"] = df["conference"].astype(str).str.slice(0, 4)

    segments = ["entropy_beginning", "entropy_middle", "entropy_end"]
    years = sorted(df["year"].dropna().unique())

    out_lines = []
    out_lines.append("CDP ENTROPY BY COHORT (PAIRWISE TESTS)\n")
    out_lines.append("=" * 80 + "\n")
    out_lines.append(f"Source: {path}\n")

    for seg in segments:
        out_lines.append(f"\n{seg.upper()}\n")
        out_lines.append("-" * 80 + "\n")
        vals = {y: df.loc[df["year"] == y, seg].dropna().values for y in years}

        for y in years:
            v = vals[y]
            out_lines.append(
                f"  {y}: n={len(v)}, mean={np.mean(v):.4f}, median={np.median(v):.4f}, "
                f"std={np.std(v, ddof=1):.4f}\n"
            )

        out_lines.append("\n  Pairwise Mann-Whitney U (two-sided)\n")

        pairs = list(combinations(years, 2))
        raw = []
        for a, b in pairs:
            va, vb = vals[a], vals[b]
            if len(va) == 0 or len(vb) == 0:
                u = np.nan
                p = np.nan
            else:
                u, p = mannwhitneyu(va, vb, alternative="two-sided")

            if len(va) == 0 or len(vb) == 0:
                rbc = np.nan
            else:
                rbc = 1 - (2 * u) / (len(va) * len(vb))

            raw.append((a, b, u, p, rbc))

        ps = [r[3] for r in raw]
        order = sorted([i for i, p in enumerate(ps) if not np.isnan(p)], key=lambda i: ps[i])
        m = len(order)
        holm = [np.nan] * len(ps)
        for rank, idx in enumerate(order, start=1):
            holm[idx] = min(ps[idx] * (m - rank + 1), 1.0)

        for i, (a, b, u, p, rbc) in enumerate(raw):
            out_lines.append(
                f"    {a} vs {b}: U={u:.1f} p={p:.4f} holm_p={holm[i]:.4f} rbc={rbc:.3f}\n"
            )

    out_path = ANALYSIS_DIR / "cdp_by_cohort_pairwise.txt"
    out_path.write_text("".join(out_lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
