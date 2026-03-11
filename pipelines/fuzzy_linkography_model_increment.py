#!/usr/bin/env python3
"""Test whether fuzzy features add predictive value on top of existing models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_gd(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, n_iter: int = 2000, lr: float = 0.1) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(n_iter):
        p = _sigmoid(x @ beta)
        grad = (x.T @ (p - y)) / len(y)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad
    return beta


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _stratified_folds(y: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos):
        folds[i % n_folds].append(int(idx))
    for i, idx in enumerate(neg):
        folds[i % n_folds].append(int(idx))
    return [np.array(sorted(f), dtype=int) for f in folds]


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train - mu) / sigma, (test - mu) / sigma


def _cv_auc(df: pd.DataFrame, feature_cols: list[str], target: str = "any_funded") -> dict[str, float]:
    sub = df[feature_cols + [target]].dropna().copy()
    if len(sub) < 20:
        return {"n": len(sub), "cv_auc_mean": float("nan"), "cv_auc_std": float("nan")}
    x = sub[feature_cols].astype(float).to_numpy()
    y = sub[target].astype(int).to_numpy()
    folds = _stratified_folds(y, n_folds=5)
    aucs = []
    for fold in folds:
        test_idx = fold
        train_mask = np.ones(len(sub), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = x[train_mask], x[test_idx]
        y_train, y_test = y[train_mask], y[test_idx]
        x_train, x_test = _standardize(x_train, x_test)
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        beta = _fit_logistic_gd(x_train, y_train)
        score = _sigmoid(x_test @ beta)
        aucs.append(_auc_score(y_test, score))
    return {
        "n": len(sub),
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_std": float(np.nanstd(aucs)),
    }


def main() -> None:
    entropy = pd.read_csv(TABLES_DIR / "entropy_with_outcomes.csv")
    speaker = pd.read_csv(TABLES_DIR / "speaker_diversity_with_outcomes.csv")
    timing = pd.read_csv(TABLES_DIR / "timing_features_with_outcomes.csv")
    fuzzy = pd.read_csv(TABLES_DIR / "fuzzy_linkography_with_outcomes_by_session.csv")

    base = entropy.copy()
    speaker_cols = ["session_id", "gini_score1", "gini_score2", "speaker_participation_cdp"]
    timing_cols = ["session_id", "purity_ratio", "mixed_ratio", "mean_jump", "max_jump", "n_transitions", "entropy_trend"]
    fuzzy_cols = ["session_id", "weighted_ldi", "mean_nonzero_weight", "cross_speaker_weight_ratio", "late_minus_early_backlink", "overall_link_entropy"]
    base = base.merge(speaker[speaker_cols], on="session_id", how="left")
    base = base.merge(timing[timing_cols], on="session_id", how="left")
    base = base.merge(fuzzy[fuzzy_cols], on="session_id", how="left")

    feature_sets = {
        "baseline_entropy": ["entropy_beginning", "entropy_middle", "entropy_end"],
        "baseline_full": [
            "entropy_beginning", "entropy_middle", "entropy_end",
            "gini_score1", "gini_score2", "speaker_participation_cdp",
            "purity_ratio", "mixed_ratio", "mean_jump", "max_jump", "n_transitions", "entropy_trend",
        ],
        "fuzzy_only": [
            "weighted_ldi", "mean_nonzero_weight", "cross_speaker_weight_ratio",
            "late_minus_early_backlink", "overall_link_entropy",
        ],
        "baseline_full_plus_fuzzy": [
            "entropy_beginning", "entropy_middle", "entropy_end",
            "gini_score1", "gini_score2", "speaker_participation_cdp",
            "purity_ratio", "mixed_ratio", "mean_jump", "max_jump", "n_transitions", "entropy_trend",
            "weighted_ldi", "mean_nonzero_weight", "cross_speaker_weight_ratio",
            "late_minus_early_backlink", "overall_link_entropy",
        ],
    }

    rows = []
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in base.columns]
        res = _cv_auc(base, cols)
        rows.append({"model": name, "n_features": len(cols), **res})

    out_df = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    out_csv = TABLES_DIR / "fuzzy_linkography_model_increment.csv"
    out_txt = ANALYSIS_DIR / "fuzzy_linkography_model_increment.txt"
    out_df.to_csv(out_csv, index=False)

    base_auc = out_df.loc[out_df["model"] == "baseline_full", "cv_auc_mean"].iloc[0]
    plus_auc = out_df.loc[out_df["model"] == "baseline_full_plus_fuzzy", "cv_auc_mean"].iloc[0]
    delta = plus_auc - base_auc

    with open(out_txt, "w") as f:
        f.write("FUZZY LINKOGRAPHY MODEL INCREMENT TEST\n")
        f.write("=" * 72 + "\n\n")
        f.write("5-fold stratified CV logistic regression implemented in pure numpy.\n\n")
        for _, row in out_df.iterrows():
            f.write(f"{row['model']}\n")
            f.write(f"  n={int(row['n'])} features={int(row['n_features'])}\n")
            f.write(f"  CV AUC: {row['cv_auc_mean']:.4f} ± {row['cv_auc_std']:.4f}\n\n")
        f.write("Increment over existing feature model:\n")
        f.write(f"  baseline_full AUC: {base_auc:.4f}\n")
        f.write(f"  baseline_full_plus_fuzzy AUC: {plus_auc:.4f}\n")
        f.write(f"  delta: {delta:.4f}\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
