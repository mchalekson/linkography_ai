#!/usr/bin/env python
"""Post-hoc analyses: effect sizes, plots, and transcript sanity checks.

Creates:
- speaker_diversity_effect_sizes.csv/.txt
- figures/final/gini_by_funding.png
- cdp_by_cohort_effect_sizes.txt
- gini_sanity_excerpts.txt
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"
FIG_DIR = REPO_ROOT / "figures" / "final"


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return low, high


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u) / (n1 * n2)


def load_session_json(conference: str, session_id: str) -> dict | None:
    session_path = DATA_DIR / conference / "session_data" / f"{session_id}.json"
    if not session_path.exists():
        return None
    try:
        return json.loads(session_path.read_text())
    except Exception:
        return None


def extract_cdp_utterances(session_json: dict) -> List[dict]:
    if isinstance(session_json, dict) and "all_data" in session_json:
        data = session_json.get("all_data", [])
    elif isinstance(session_json, list):
        data = session_json
    else:
        data = []

    utterances = []
    for u in data:
        if not isinstance(u, dict):
            continue
        ann = u.get("annotations") or u.get("annotation_dict") or {}
        if not isinstance(ann, dict):
            continue
        cdp = ann.get("Coordination and Decision Practices")
        if not isinstance(cdp, dict):
            continue
        score = cdp.get("score")
        if score not in (1, 2):
            continue
        text = (u.get("transcript") or u.get("text") or u.get("utterance") or "").strip()
        if not text:
            continue
        utterances.append(
            {
                "speaker": (u.get("speaker") or "").strip(),
                "score": int(score),
                "text": text,
                "tokens": len(text.split()),
            }
        )
    return utterances


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Speaker diversity effect sizes
    speaker_path = TABLES_DIR / "speaker_diversity_with_outcomes.csv"
    if not speaker_path.exists():
        print("Missing speaker_diversity_with_outcomes.csv. Run speaker_diversity_outcomes.py first.")
        return

    df = pd.read_csv(speaker_path)
    funded = df[df["any_funded"] == 1]
    unfunded = df[df["any_funded"] == 0]

    metrics = ["gini_score1", "gini_score2", "speaker_participation_cdp"]
    rows = []
    for m in metrics:
        a = funded[m].dropna().to_numpy()
        b = unfunded[m].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        low, high = bootstrap_mean_diff(a, b)
        rows.append(
            {
                "metric": m,
                "funded_mean": a.mean(),
                "unfunded_mean": b.mean(),
                "mean_diff": a.mean() - b.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "cohens_d": cohens_d(a, b),
            }
        )

    eff_df = pd.DataFrame(rows)
    eff_df.to_csv(TABLES_DIR / "speaker_diversity_effect_sizes.csv", index=False)

    with open(ANALYSIS_DIR / "speaker_diversity_effect_sizes.txt", "w") as f:
        f.write("SPEAKER DIVERSITY EFFECT SIZES (FUNDED - UNFUNDED)\n")
        f.write("=" * 80 + "\n\n")
        for _, r in eff_df.iterrows():
            f.write(
                f"{r['metric']}: diff={r['mean_diff']:.3f} (95% CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]), d={r['cohens_d']:.3f}\n"
            )

    # Plot Gini by funding
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    data = [
        [funded["gini_score1"].dropna(), unfunded["gini_score1"].dropna()],
        [funded["gini_score2"].dropna(), unfunded["gini_score2"].dropna()],
    ]
    titles = ["Gini Score 1", "Gini Score 2"]
    for i in range(2):
        ax[i].boxplot(data[i], tick_labels=["Funded", "Unfunded"], showfliers=False)
        ax[i].set_title(titles[i])
        ax[i].set_ylim(0, 1)
    fig.suptitle("Coordination Concentration by Funding")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gini_by_funding.png", dpi=200)

    # Cohort effect sizes (pairwise rank-biserial from U)
    entropy_files = sorted(TABLES_DIR.glob("cdp_entropy_by_session_ALL_*.csv"))
    if entropy_files:
        entropy_df = pd.read_csv(entropy_files[-1])
        entropy_df["year"] = entropy_df["conference"].astype(str).str.slice(0, 4)
        years = sorted(entropy_df["year"].dropna().unique())
        segments = ["entropy_beginning", "entropy_middle", "entropy_end"]

        lines = []
        lines.append("CDP COHORT EFFECT SIZES (RANK-BISERIAL)\n")
        lines.append("=" * 80 + "\n")
        lines.append(f"Source: {entropy_files[-1]}\n\n")
        for seg in segments:
            lines.append(f"{seg}\n")
            lines.append("-" * 80 + "\n")
            for i in range(len(years)):
                for j in range(i + 1, len(years)):
                    a = entropy_df.loc[entropy_df["year"] == years[i], seg].dropna().to_numpy()
                    b = entropy_df.loc[entropy_df["year"] == years[j], seg].dropna().to_numpy()
                    if len(a) == 0 or len(b) == 0:
                        continue
                    # Mann-Whitney U
                    from scipy.stats import mannwhitneyu
                    u, _ = mannwhitneyu(a, b, alternative="two-sided")
                    rbc = rank_biserial_from_u(u, len(a), len(b))
                    lines.append(f"  {years[i]} vs {years[j]}: rbc={rbc:.3f}\n")
            lines.append("\n")

        (ANALYSIS_DIR / "cdp_by_cohort_effect_sizes.txt").write_text("".join(lines))

    # Transcript sanity check: 2 high-Gini funded vs 2 low-Gini unfunded
    sanity_path = ANALYSIS_DIR / "gini_sanity_excerpts.txt"
    merged = df.copy()
    high_funded = merged[merged["any_funded"] == 1].sort_values("gini_score2", ascending=False).head(2)
    low_unfunded = merged[merged["any_funded"] == 0].sort_values("gini_score2", ascending=True).head(2)

    with open(sanity_path, "w") as f:
        f.write("GINI SANITY CHECK EXCERPTS (FUNDED HIGH vs UNFUNDED LOW)\n")
        f.write("=" * 80 + "\n\n")
        for label, rowset in [("FUNDED_HIGH_GINI", high_funded), ("UNFUNDED_LOW_GINI", low_unfunded)]:
            f.write(f"[{label}]\n")
            f.write("-" * 80 + "\n")
            for _, row in rowset.iterrows():
                conference = row["conference"]
                session_id = row["session_id"]
                session_json = load_session_json(conference, session_id)
                if not session_json:
                    continue
                utterances = extract_cdp_utterances(session_json)
                # pick top 3 longest score-2 utterances
                s2 = [u for u in utterances if u["score"] == 2]
                s2.sort(key=lambda x: x["tokens"], reverse=True)
                f.write(f"Session: {session_id}\n")
                for u in s2[:3]:
                    f.write(f"  (score 2, {u['tokens']} tokens) {u['speaker']}: {u['text']}\n")
                f.write("\n")

    print("Saved post-hoc outputs:")
    print("- outputs/tables/speaker_diversity_effect_sizes.csv")
    print("- outputs/analysis/speaker_diversity_effect_sizes.txt")
    print("- figures/final/gini_by_funding.png")
    print("- outputs/analysis/cdp_by_cohort_effect_sizes.txt")
    print("- outputs/analysis/gini_sanity_excerpts.txt")


if __name__ == "__main__":
    main()
