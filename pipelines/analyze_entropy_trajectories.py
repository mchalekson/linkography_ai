#!/usr/bin/env python
"""Entropy Trajectory Analysis

Analyzes entropy trajectories (beginning → middle → end) from batch entropy output.
Computes summary statistics, paired comparisons, and visualizations.

Usage:
    python pipelines/analyze_entropy_trajectories.py
    python pipelines/analyze_entropy_trajectories.py --csv outputs/tables/cdp_entropy_by_session_ALL_*.csv
"""

from __future__ import annotations

import argparse
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
    """Find the most recent entropy CSV in outputs/tables/."""
    if not TABLES_DIR.exists():
        return None
    
    csv_files = list(TABLES_DIR.glob("cdp_entropy_by_session_*.csv"))
    if not csv_files:
        return None
    
    # Sort by modification time
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def bootstrap_paired_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for paired difference (x - y).
    
    Returns:
        dict with mean_diff, lower_ci, upper_ci
    """
    n = len(x)
    diffs = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_diff = np.mean(x[idx] - y[idx])
        diffs.append(boot_diff)
    
    diffs = np.array(diffs)
    alpha = 1 - ci
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    
    return {
        "mean_diff": np.mean(x - y),
        "lower_ci": lower,
        "upper_ci": upper
    }


def analyze_trajectories(df: pd.DataFrame) -> dict:
    """Compute trajectory statistics.
    
    Args:
        df: DataFrame with entropy_beginning, entropy_middle, entropy_end columns
    
    Returns:
        dict with summary stats and paired comparisons
    """
    # Filter out NaN values
    valid = df[["entropy_beginning", "entropy_middle", "entropy_end"]].dropna()
    n_valid = len(valid)
    
    if n_valid == 0:
        return {"error": "No valid entropy values found"}
    
    beginning = valid["entropy_beginning"].values
    middle = valid["entropy_middle"].values
    end = valid["entropy_end"].values
    
    # Summary stats
    summary = {
        "n_sessions": n_valid,
        "beginning": {
            "mean": float(np.mean(beginning)),
            "median": float(np.median(beginning)),
            "std": float(np.std(beginning, ddof=1)),
        },
        "middle": {
            "mean": float(np.mean(middle)),
            "median": float(np.median(middle)),
            "std": float(np.std(middle, ddof=1)),
        },
        "end": {
            "mean": float(np.mean(end)),
            "median": float(np.median(end)),
            "std": float(np.std(end, ddof=1)),
        }
    }
    
    # Paired comparisons with bootstrap CIs
    summary["beginning_vs_middle"] = bootstrap_paired_diff(beginning, middle)
    summary["middle_vs_end"] = bootstrap_paired_diff(middle, end)
    summary["beginning_vs_end"] = bootstrap_paired_diff(beginning, end)
    
    return summary


def plot_trajectories(df: pd.DataFrame, output_path: Path) -> None:
    """Create trajectory visualization."""
    valid = df[["entropy_beginning", "entropy_middle", "entropy_end"]].dropna()
    
    if len(valid) == 0:
        print("No valid data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Mean with error bars
    means = [
        valid["entropy_beginning"].mean(),
        valid["entropy_middle"].mean(),
        valid["entropy_end"].mean()
    ]
    
    stds = [
        valid["entropy_beginning"].std(ddof=1),
        valid["entropy_middle"].std(ddof=1),
        valid["entropy_end"].std(ddof=1)
    ]
    
    sems = [s / np.sqrt(len(valid)) for s in stds]
    
    phases = ["Beginning", "Middle", "End"]
    x_pos = np.arange(len(phases))
    
    axes[0].bar(x_pos, means, yerr=sems, capsize=5, alpha=0.7, color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(phases)
    axes[0].set_ylabel("Mean Entropy")
    axes[0].set_title(f"Entropy Trajectory (n={len(valid)} sessions)")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right plot: Individual trajectories (sample if > 50)
    n_sample = min(50, len(valid))
    sample = valid.sample(n=n_sample, random_state=42) if len(valid) > 50 else valid
    
    for idx, row in sample.iterrows():
        axes[1].plot([0, 1, 2], 
                     [row["entropy_beginning"], row["entropy_middle"], row["entropy_end"]],
                     alpha=0.2, color="gray", linewidth=0.5)
    
    # Add mean line
    axes[1].plot([0, 1, 2], means, color="red", linewidth=2, marker='o', label="Mean")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(phases)
    axes[1].set_ylabel("Entropy")
    axes[1].set_title(f"Individual Trajectories (showing {n_sample})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {output_path}")
    plt.close()


def write_summary(stats: dict, output_path: Path) -> None:
    """Write summary statistics to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ENTROPY TRAJECTORY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        if "error" in stats:
            f.write(f"ERROR: {stats['error']}\n")
            return
        
        f.write(f"Number of sessions: {stats['n_sessions']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS BY PHASE\n")
        f.write("-" * 80 + "\n\n")
        
        for phase in ["beginning", "middle", "end"]:
            f.write(f"{phase.upper()}:\n")
            f.write(f"  Mean:   {stats[phase]['mean']:.4f}\n")
            f.write(f"  Median: {stats[phase]['median']:.4f}\n")
            f.write(f"  Std:    {stats[phase]['std']:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PAIRED COMPARISONS (with 95% Bootstrap CIs)\n")
        f.write("-" * 80 + "\n\n")
        
        comparisons = [
            ("Beginning vs Middle", "beginning_vs_middle"),
            ("Middle vs End", "middle_vs_end"),
            ("Beginning vs End", "beginning_vs_end")
        ]
        
        for label, key in comparisons:
            comp = stats[key]
            f.write(f"{label}:\n")
            f.write(f"  Mean difference: {comp['mean_diff']:.4f}\n")
            f.write(f"  95% CI: [{comp['lower_ci']:.4f}, {comp['upper_ci']:.4f}]\n")
            
            # Interpretation
            if comp['lower_ci'] > 0:
                f.write(f"  → Significantly HIGHER in first phase (CI excludes 0)\n")
            elif comp['upper_ci'] < 0:
                f.write(f"  → Significantly LOWER in first phase (CI excludes 0)\n")
            else:
                f.write(f"  → No significant difference (CI includes 0)\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n\n")
        
        beg_end_diff = stats["beginning_vs_end"]["mean_diff"]
        if beg_end_diff > 0.1 and stats["beginning_vs_end"]["lower_ci"] > 0:
            f.write("✅ Entropy DECREASES from beginning to end (convergence pattern)\n")
        elif beg_end_diff < -0.1 and stats["beginning_vs_end"]["upper_ci"] < 0:
            f.write("⚠️ Entropy INCREASES from beginning to end (divergence pattern)\n")
        else:
            f.write("➖ Entropy shows no clear trajectory (stable or inconsistent)\n")


def main() -> None:
    """Run trajectory analysis."""
    parser = argparse.ArgumentParser(description="Analyze entropy trajectories")
    parser.add_argument("--csv", type=Path, help="Path to entropy CSV (auto-detects if not provided)")
    args = parser.parse_args()
    
    # Find CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_entropy_csv()
    
    if csv_path is None or not csv_path.exists():
        print("ERROR: No entropy CSV found.")
        print("Run this first:")
        print("  python pipelines/run_cdp_entropy_all.py --conference ALL --normalize")
        return
    
    print(f"Loading entropy data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} sessions")
    print()
    
    # Create output directories
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze
    print("Computing trajectory statistics...")
    stats = analyze_trajectories(df)
    
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return
    
    # Write summary
    summary_path = ANALYSIS_DIR / "entropy_trajectory_summary.txt"
    write_summary(stats, summary_path)
    print(f"Saved summary: {summary_path}")
    
    # Plot
    print("Generating visualization...")
    plot_path = FIGURES_DIR / "entropy_trajectory.png"
    plot_trajectories(df, plot_path)
    
    print()
    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Sessions analyzed: {stats['n_sessions']}")
    print(f"Beginning entropy: {stats['beginning']['mean']:.3f} ± {stats['beginning']['std']:.3f}")
    print(f"Middle entropy:    {stats['middle']['mean']:.3f} ± {stats['middle']['std']:.3f}")
    print(f"End entropy:       {stats['end']['mean']:.3f} ± {stats['end']['std']:.3f}")
    print()
    
    beg_end = stats["beginning_vs_end"]
    print(f"Beginning → End change: {beg_end['mean_diff']:.3f} [{beg_end['lower_ci']:.3f}, {beg_end['upper_ci']:.3f}]")
    
    if beg_end['lower_ci'] > 0:
        print("✅ Significant DECREASE (convergence)")
    elif beg_end['upper_ci'] < 0:
        print("⚠️ Significant INCREASE (divergence)")
    else:
        print("➖ No significant change")


if __name__ == "__main__":
    main()
