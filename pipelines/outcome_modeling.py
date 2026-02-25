#!/usr/bin/env python
"""Outcome modeling beyond entropy.

Fits simple linear models predicting funded_rate and any_funded using
entropy and additional signals (convergence, structural wrap, time pressure).
Outputs coefficient table and report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def find_entropy_with_outcomes_csv() -> Optional[Path]:
    path = TABLES_DIR / "entropy_with_outcomes.csv"
    return path if path.exists() else None


def load_optional_csv(name: str) -> Optional[pd.DataFrame]:
    path = TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def fit_linear_model(df: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    sub = df[[target] + predictors].dropna()
    y = sub[target].astype(float).values
    X = sub[predictors].astype(float).values
    X = np.column_stack([np.ones(len(X)), X])

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n": len(sub),
        "coeffs": coeffs,
        "r2": r2,
        "predictors": ["intercept"] + predictors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Outcome modeling beyond entropy")
    parser.add_argument("--csv", type=Path, help="Path to entropy_with_outcomes.csv")
    args = parser.parse_args()

    csv_path = args.csv or find_entropy_with_outcomes_csv()
    if csv_path is None or not csv_path.exists():
        print("ERROR: entropy_with_outcomes.csv not found.")
        return

    entropy = pd.read_csv(csv_path)

    convergence = load_optional_csv("convergence_rates_by_session.csv")
    time_pressure = load_optional_csv("time_pressure_language_by_session.csv")

    df = entropy.copy()
    if convergence is not None:
        df = df.merge(convergence, on=["session_id", "conference"], how="left")
    if time_pressure is not None:
        df = df.merge(time_pressure, on=["session_id", "conference"], how="left")

    df["entropy_change"] = df["entropy_end"] - df["entropy_beginning"]

    predictors = [
        "entropy_end",
        "entropy_change",
        "strict_conv_rate_last_third",
        "structural_wrap_rate_last_third",
        "time_pressure_total",
        "decision_closure_total",
    ]

    # Ensure predictors exist
    predictors = [p for p in predictors if p in df.columns]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = ANALYSIS_DIR / "outcome_modeling_report.txt"
    coeffs_path = TABLES_DIR / "outcome_model_coefficients.csv"

    results = []

    with open(report_path, "w") as f:
        f.write("OUTCOME MODELING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Predictors used: {', '.join(predictors)}\n\n")

        for target in ["funded_rate", "any_funded"]:
            if target not in df.columns:
                continue
            model = fit_linear_model(df, target, predictors)
            f.write(f"Target: {target}\n")
            f.write(f"n = {model['n']}\n")
            f.write(f"R^2 = {model['r2']:.4f}\n")
            f.write("Coefficients:\n")
            for name, coef in zip(model["predictors"], model["coeffs"]):
                f.write(f"  {name}: {coef:.4f}\n")
            f.write("\n")

            for name, coef in zip(model["predictors"], model["coeffs"]):
                results.append({
                    "target": target,
                    "predictor": name,
                    "coef": float(coef),
                    "n": model["n"],
                    "r2": model["r2"],
                })

    pd.DataFrame(results).to_csv(coeffs_path, index=False)

    print(f"Saved: {report_path}")
    print(f"Saved: {coeffs_path}")


if __name__ == "__main__":
    main()
