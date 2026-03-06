#!/usr/bin/env python
"""First-pass outcomes analysis for fuzzy linkography features.

Uses session-level fuzzy features merged with outcomes to run:
  - Spearman correlations vs any_funded and funded_rate
  - Mann-Whitney tests (funded vs unfunded)
  - Cohen's d effect size
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    dof = nx + ny - 2
    if dof <= 0:
        return float("nan")
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof
    if pooled_var <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / (pooled_var ** 0.5))


def main() -> None:
    in_path = TABLES_DIR / "fuzzy_linkography_with_outcomes_by_session.csv"
    if not in_path.exists():
        print(f"ERROR: Missing input file: {in_path}")
        print("Run: conda run -n gem_samp python pipelines/merge_fuzzy_with_outcomes.py")
        return

    df = pd.read_csv(in_path)

    # Only rows with outcomes for inferential tests
    modeled = df[df["any_funded"].notna()].copy()
    if modeled.empty:
        print("No sessions with outcomes available.")
        return

    modeled["any_funded"] = modeled["any_funded"].astype(int)

    features = [
        "weighted_ldi",
        "mean_nonzero_weight",
        "cross_speaker_weight_ratio",
        "late_minus_early_backlink",
        "overall_link_entropy",
        "forelink_entropy",
        "backlink_entropy",
        "horizon_entropy",
    ]
    features = [f for f in features if f in modeled.columns]

    funded = modeled[modeled["any_funded"] == 1]
    unfunded = modeled[modeled["any_funded"] == 0]

    rows = []
    for feat in features:
        x = modeled[feat].astype(float)
        y_any = modeled["any_funded"].astype(float)
        corr_any, p_corr_any = spearmanr(x, y_any, nan_policy="omit")

        corr_rate, p_corr_rate = (float("nan"), float("nan"))
        if "funded_rate" in modeled.columns:
            y_rate = modeled["funded_rate"].astype(float)
            corr_rate, p_corr_rate = spearmanr(x, y_rate, nan_policy="omit")

        fvals = funded[feat].dropna().astype(float)
        uvals = unfunded[feat].dropna().astype(float)
        if len(fvals) > 0 and len(uvals) > 0:
            stat, p_mw = mannwhitneyu(fvals, uvals)
        else:
            stat, p_mw = float("nan"), float("nan")

        rows.append(
            {
                "feature": feat,
                "funded_mean": float(fvals.mean()) if len(fvals) else float("nan"),
                "unfunded_mean": float(uvals.mean()) if len(uvals) else float("nan"),
                "mannwhitney_u": float(stat),
                "p_mannwhitney": float(p_mw),
                "cohens_d": cohens_d(fvals, uvals),
                "spearman_r_any_funded": float(corr_any),
                "p_spearman_any_funded": float(p_corr_any),
                "spearman_r_funded_rate": float(corr_rate),
                "p_spearman_funded_rate": float(p_corr_rate),
                "n_funded": int(len(fvals)),
                "n_unfunded": int(len(uvals)),
            }
        )

    res = pd.DataFrame(rows).sort_values("p_mannwhitney", na_position="last")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = TABLES_DIR / "fuzzy_linkography_outcomes_tests.csv"
    out_txt = ANALYSIS_DIR / "fuzzy_linkography_outcomes_summary.txt"
    res.to_csv(out_csv, index=False)

    with open(out_txt, "w") as f:
        f.write("FUZZY LINKOGRAPHY OUTCOMES ANALYSIS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Sessions with outcomes: {len(modeled)}\n")
        f.write(f"Funded sessions: {len(funded)}\n")
        f.write(f"Unfunded sessions: {len(unfunded)}\n\n")
        f.write("Feature tests (sorted by Mann-Whitney p):\n\n")
        for _, r in res.iterrows():
            f.write(f"{r['feature']}\n")
            f.write(f"  Funded mean:   {r['funded_mean']:.4f}\n")
            f.write(f"  Unfunded mean: {r['unfunded_mean']:.4f}\n")
            f.write(f"  Mann-Whitney p: {r['p_mannwhitney']:.4f}\n")
            f.write(f"  Cohen's d: {r['cohens_d']:.4f}\n")
            f.write(f"  Spearman(any_funded): r={r['spearman_r_any_funded']:.4f}, p={r['p_spearman_any_funded']:.4f}\n")
            f.write(f"  Spearman(funded_rate): r={r['spearman_r_funded_rate']:.4f}, p={r['p_spearman_funded_rate']:.4f}\n")
            f.write("\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
