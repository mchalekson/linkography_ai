#!/usr/bin/env python3
"""Conference-specific robustness checks for fuzzy linkography features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def _rankdata(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    xr = _rankdata(x[mask].astype(float))
    yr = _rankdata(y[mask].astype(float))
    return float(xr.corr(yr))


def _cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    dof = nx + ny - 2
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof
    if pooled_var <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / (pooled_var ** 0.5))


def main() -> None:
    in_path = TABLES_DIR / "fuzzy_linkography_with_outcomes_by_session.csv"
    df = pd.read_csv(in_path)
    df = df[df["any_funded"].notna()].copy()
    if df.empty:
        print("No sessions with outcomes available.")
        return

    df["any_funded"] = df["any_funded"].astype(int)
    features = [
        "mean_nonzero_weight",
        "weighted_ldi",
        "cross_speaker_weight_ratio",
        "late_minus_early_backlink",
        "overall_link_entropy",
    ]
    features = [f for f in features if f in df.columns]

    rows = []
    for conference, g in df.groupby("conference"):
        funded = g[g["any_funded"] == 1]
        unfunded = g[g["any_funded"] == 0]
        for feature in features:
            fvals = funded[feature].dropna().astype(float)
            uvals = unfunded[feature].dropna().astype(float)
            row = {
                "conference": conference,
                "feature": feature,
                "n_sessions": int(len(g)),
                "n_funded": int(len(funded)),
                "n_unfunded": int(len(unfunded)),
                "funded_mean": float(fvals.mean()) if len(fvals) else float("nan"),
                "unfunded_mean": float(uvals.mean()) if len(uvals) else float("nan"),
                "mean_diff": float(fvals.mean() - uvals.mean()) if len(fvals) and len(uvals) else float("nan"),
                "cohens_d": _cohens_d(fvals, uvals),
                "spearman_any_funded": _spearman(g[feature], g["any_funded"]),
                "spearman_funded_rate": _spearman(g[feature], g["funded_rate"]) if "funded_rate" in g.columns else float("nan"),
            }
            rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["feature", "conference"]).reset_index(drop=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = TABLES_DIR / "fuzzy_linkography_conference_robustness.csv"
    out_txt = ANALYSIS_DIR / "fuzzy_linkography_conference_robustness.txt"
    out_df.to_csv(out_csv, index=False)

    focus = out_df[out_df["feature"] == "mean_nonzero_weight"].copy()
    with open(out_txt, "w") as f:
        f.write("FUZZY LINKOGRAPHY CONFERENCE ROBUSTNESS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Input rows with outcomes: {len(df)}\n")
        if "similarity_method" in df.columns and df["similarity_method"].notna().any():
            f.write(f"Similarity method: {df['similarity_method'].dropna().iloc[0]}\n")
        f.write("\nConference-level summary for mean_nonzero_weight:\n\n")
        for _, row in focus.iterrows():
            f.write(f"{row['conference']}\n")
            f.write(f"  Sessions: {int(row['n_sessions'])} | funded={int(row['n_funded'])} unfunded={int(row['n_unfunded'])}\n")
            f.write(f"  Funded mean:   {row['funded_mean']:.4f}\n")
            f.write(f"  Unfunded mean: {row['unfunded_mean']:.4f}\n")
            f.write(f"  Mean diff:     {row['mean_diff']:.4f}\n")
            f.write(f"  Cohen's d:     {row['cohens_d']:.4f}\n")
            f.write(f"  Spearman(any_funded): {row['spearman_any_funded']:.4f}\n")
            f.write(f"  Spearman(funded_rate): {row['spearman_funded_rate']:.4f}\n\n")

        f.write("Interpretation:\n")
        f.write("  Look for direction consistency rather than significance alone.\n")
        f.write("  If funded < unfunded repeats across conferences, the first-pass signal is directionally robust.\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
