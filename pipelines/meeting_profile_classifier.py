#!/usr/bin/env python
"""Meeting Profile Classifier

Builds predictive models to classify funding outcomes using speaker diversity
and timing features. Compares to entropy-only baseline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all needed data
    try:
        outcomes_df = pd.read_csv(TABLES_DIR / "entropy_with_outcomes.csv")
        speaker_df = pd.read_csv(TABLES_DIR / "speaker_diversity_with_outcomes.csv")
        timing_df = pd.read_csv(TABLES_DIR / "timing_features_with_outcomes.csv")
    except FileNotFoundError as e:
        print(f"ERROR: Missing required file: {e}")
        print("Ensure speaker_diversity_outcomes.py and timing_patterns_outcomes.py have been run.")
        return

    # Merge all data
    # Start with outcomes as base
    combined = outcomes_df.copy()
    
    # Add speaker diversity features
    speaker_features = ["gini_score1", "gini_score2", "speaker_participation_cdp"]
    speaker_subset = speaker_df[["session_id"] + speaker_features]
    combined = combined.merge(speaker_subset, on="session_id", how="left")
    
    # Add timing features
    timing_features = ["purity_ratio", "mixed_ratio", "mean_jump", "max_jump", "n_transitions", "entropy_trend"]
    timing_subset = timing_df[["session_id"] + timing_features]
    combined = combined.merge(timing_subset, on="session_id", how="left")

    # Keep only complete cases
    required_features = (
        ["entropy_beginning", "entropy_middle", "entropy_end"] +
        ["gini_score1", "gini_score2", "speaker_participation_cdp"] +
        timing_features
    )
    combined_clean = combined.dropna(subset=required_features + ["any_funded"])

    if len(combined_clean) < 10:
        print(f"Not enough complete cases: {len(combined_clean)}")
        return

    X = combined_clean[required_features]
    y = combined_clean["any_funded"].values

    # Feature groups for comparison
    entropy_only = ["entropy_beginning", "entropy_middle", "entropy_end"]
    speaker_only = ["gini_score1", "gini_score2", "speaker_participation_cdp"]
    timing_only = timing_features
    combined_features = entropy_only + ["gini_score1", "gini_score2", "speaker_participation_cdp"] + timing_features

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models on each feature group
    models_to_test = [
        ("Entropy Only (Baseline)", entropy_only),
        ("Speaker Diversity Only", speaker_only),
        ("Timing Patterns Only", timing_only),
        ("All Features Combined", combined_features),
    ]

    results = []

    report_path = ANALYSIS_DIR / "meeting_profile_classifier_results.txt"
    with open(report_path, "w") as f:
        f.write("MEETING PROFILE CLASSIFIER: PREDICTING FUNDING OUTCOMES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sessions with complete data: {len(combined_clean)}\n")
        f.write(f"Funded: {y.sum()} ({100*y.sum()/len(y):.1f}%)\n")
        f.write(f"Unfunded: {len(y) - y.sum()} ({100*(len(y)-y.sum())/len(y):.1f}%)\n\n")

        for model_name, features in models_to_test:
            f.write(f"\n{model_name.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Features: {', '.join(features)}\n")
            f.write(f"N features: {len(features)}\n\n")

            # Get feature indices
            feature_indices = [required_features.index(feat) for feat in features if feat in required_features]
            X_subset = X_scaled[:, feature_indices]

            # Logistic Regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr_scores = cross_val_score(lr, X_subset, y, cv=5, scoring="roc_auc")
            
            lr.fit(X_subset, y)
            lr_auc = roc_auc_score(y, lr.predict_proba(X_subset)[:, 1])
            lr_f1 = f1_score(y, lr.predict(X_subset))

            f.write("Logistic Regression (5-fold cross-validation):\n")
            f.write(f"  ROC-AUC CV: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}\n")
            f.write(f"  ROC-AUC (fit): {lr_auc:.3f}\n")
            f.write(f"  F1-Score: {lr_f1:.3f}\n\n")

            results.append({
                "Model": "Logistic Regression",
                "Features": model_name,
                "CV_AUC": lr_scores.mean(),
                "Fit_AUC": lr_auc,
                "F1": lr_f1,
            })

            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf_scores = cross_val_score(rf, X_subset, y, cv=5, scoring="roc_auc")
            
            rf.fit(X_subset, y)
            rf_auc = roc_auc_score(y, rf.predict_proba(X_subset)[:, 1])
            rf_f1 = f1_score(y, rf.predict(X_subset))

            f.write("Random Forest (5-fold cross-validation):\n")
            f.write(f"  ROC-AUC CV: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}\n")
            f.write(f"  ROC-AUC (fit): {rf_auc:.3f}\n")
            f.write(f"  F1-Score: {rf_f1:.3f}\n\n")

            results.append({
                "Model": "Random Forest",
                "Features": model_name,
                "CV_AUC": rf_scores.mean(),
                "Fit_AUC": rf_auc,
                "F1": rf_f1,
            })

            # Feature importance
            if model_name != "Entropy Only (Baseline)":
                importances = rf.feature_importances_
                feature_importance_pairs = list(zip(features, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                f.write("Random Forest Feature Importance:\n")
                for feat, imp in feature_importance_pairs:
                    f.write(f"  {feat}: {imp:.3f}\n")
                f.write("\n")

        # Summary comparison
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY: WHICH FEATURES PREDICT FUNDING BEST?\n")
        f.write("=" * 80 + "\n\n")

        results_df = pd.DataFrame(results)
        
        # Get best by CV AUC
        best_lr = results_df[results_df["Model"] == "Logistic Regression"].nlargest(1, "CV_AUC").iloc[0]
        best_rf = results_df[results_df["Model"] == "Random Forest"].nlargest(1, "CV_AUC").iloc[0]

        f.write("BEST LOGISTIC REGRESSION:\n")
        f.write(f"  Features: {best_lr['Features']}\n")
        f.write(f"  CV-AUC: {best_lr['CV_AUC']:.3f}\n\n")

        f.write("BEST RANDOM FOREST:\n")
        f.write(f"  Features: {best_rf['Features']}\n")
        f.write(f"  CV-AUC: {best_rf['CV_AUC']:.3f}\n\n")

        # Interpretation
        entropy_baseline = results_df[
            (results_df["Model"] == "Logistic Regression") &
            (results_df["Features"] == "Entropy Only (Baseline)")
        ].iloc[0]["CV_AUC"]

        combined_lr = results_df[
            (results_df["Model"] == "Logistic Regression") &
            (results_df["Features"] == "All Features Combined")
        ].iloc[0]["CV_AUC"]

        improvement = (combined_lr - entropy_baseline) / entropy_baseline * 100

        f.write("KEY FINDING:\n")
        f.write(f"  Entropy-only baseline (LR): {entropy_baseline:.3f}\n")
        f.write(f"  Combined features (LR): {combined_lr:.3f}\n")
        if improvement > 5:
            f.write(f"  → {improvement:.1f}% IMPROVEMENT with speaker + timing features ✓\n")
            f.write(f"  → Speaker diversity and meeting rhythm MATTER for predicting outcomes\n")
        elif improvement > -5:
            f.write(f"  → ±{abs(improvement):.1f}% no meaningful change\n")
            f.write(f"  → Speaker diversity and timing do not substantially improve prediction\n")
        else:
            f.write(f"  → {improvement:.1f}% WORSE with additional features\n")
            f.write(f"  → Entropy already captures the signal; other features add noise\n")

    # Save results table
    results_df.to_csv(TABLES_DIR / "meeting_profile_classifier_results.csv", index=False)

    print(f"Saved: {report_path}")
    print(f"Saved: {TABLES_DIR / 'meeting_profile_classifier_results.csv'}")


if __name__ == "__main__":
    main()
