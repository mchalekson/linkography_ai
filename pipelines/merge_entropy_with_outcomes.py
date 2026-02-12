#!/usr/bin/env python
"""Merge Entropy with Outcomes

Extracts team funding outcomes from *_session_outcomes.json files and merges
with entropy data. Computes funded_rate and any_funded for each session.

Usage:
    python pipelines/merge_entropy_with_outcomes.py
    python pipelines/merge_entropy_with_outcomes.py --csv outputs/tables/cdp_entropy_by_session_ALL_*.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
LOGS_DIR = OUT_DIR / "logs"


def discover_conferences() -> list[str]:
    """Find all conference directories with session_data subdirs."""
    conferences = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "session_data").exists():
            conferences.append(p.name)
    return conferences


def extract_outcomes(conference: str) -> Dict[str, dict]:
    """Extract outcomes for a conference.
    
    Args:
        conference: Conference code (e.g., '2021NES')
    
    Returns:
        dict mapping session_id to outcome dict with:
            - funded_rate: mean of funded_status across teams
            - any_funded: 1 if any team funded, 0 otherwise
            - n_teams: number of teams
    """
    outcomes_path = DATA_DIR / conference / f"{conference}_session_outcomes.json"
    
    if not outcomes_path.exists():
        return {}
    
    try:
        data = json.loads(outcomes_path.read_text())
    except Exception as e:
        print(f"Warning: Could not parse {outcomes_path}: {e}")
        return {}
    
    results = {}
    
    for session_id, session_data in data.items():
        if not isinstance(session_data, dict):
            continue
        
        teams = session_data.get("teams", {})
        if not teams:
            continue
        
        funded_statuses = []
        for team_id, team_data in teams.items():
            if isinstance(team_data, dict) and "funded_status" in team_data:
                funded_statuses.append(int(team_data["funded_status"]))
        
        if funded_statuses:
            results[session_id] = {
                "funded_rate": sum(funded_statuses) / len(funded_statuses),
                "any_funded": 1 if any(funded_statuses) else 0,
                "n_teams": len(funded_statuses)
            }
    
    return results


def find_latest_entropy_csv() -> Optional[Path]:
    """Find the most recent entropy CSV in outputs/tables/."""
    if not TABLES_DIR.exists():
        return None
    
    csv_files = list(TABLES_DIR.glob("cdp_entropy_by_session_*.csv"))
    if not csv_files:
        return None
    
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def merge_outcomes_with_entropy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge outcome data with entropy DataFrame.
    
    Args:
        df: Entropy DataFrame with 'conference' and 'session_id' columns
    
    Returns:
        (merged_df, stats_dict)
    """
    # Collect all outcomes
    print("Extracting outcomes from session_outcomes.json files...")
    all_outcomes = {}
    conferences = discover_conferences()
    
    for conf in conferences:
        outcomes = extract_outcomes(conf)
        print(f"  {conf}: {len(outcomes)} sessions with outcome data")
        all_outcomes.update(outcomes)
    
    print()
    
    # Add outcome columns
    df["funded_rate"] = None
    df["any_funded"] = None
    df["n_teams"] = None
    
    matched = 0
    unmatched_entropy = []
    unmatched_outcomes = list(all_outcomes.keys())
    
    for idx, row in df.iterrows():
        session_id = row["session_id"]
        
        if session_id in all_outcomes:
            outcome = all_outcomes[session_id]
            df.at[idx, "funded_rate"] = outcome["funded_rate"]
            df.at[idx, "any_funded"] = outcome["any_funded"]
            df.at[idx, "n_teams"] = outcome["n_teams"]
            matched += 1
            if session_id in unmatched_outcomes:
                unmatched_outcomes.remove(session_id)
        else:
            unmatched_entropy.append(session_id)
    
    stats = {
        "total_entropy_sessions": len(df),
        "total_outcome_sessions": len(all_outcomes),
        "matched": matched,
        "unmatched_entropy": unmatched_entropy,
        "unmatched_outcomes": unmatched_outcomes
    }
    
    return df, stats


def write_merge_report(stats: dict, output_path: Path) -> None:
    """Write merge statistics report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("OUTCOME MERGE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total sessions in entropy CSV: {stats['total_entropy_sessions']}\n")
        f.write(f"Total sessions with outcomes:   {stats['total_outcome_sessions']}\n")
        f.write(f"Successfully matched:           {stats['matched']}\n")
        f.write(f"Unmatched (entropy only):       {len(stats['unmatched_entropy'])}\n")
        f.write(f"Unmatched (outcome only):       {len(stats['unmatched_outcomes'])}\n\n")
        
        if stats["unmatched_entropy"]:
            f.write("-" * 80 + "\n")
            f.write("SESSIONS WITH ENTROPY BUT NO OUTCOME DATA\n")
            f.write("-" * 80 + "\n")
            for sid in stats["unmatched_entropy"]:
                f.write(f"  {sid}\n")
            f.write("\n")
        
        if stats["unmatched_outcomes"]:
            f.write("-" * 80 + "\n")
            f.write("SESSIONS WITH OUTCOME DATA BUT NO ENTROPY\n")
            f.write("-" * 80 + "\n")
            for sid in stats["unmatched_outcomes"]:
                f.write(f"  {sid}\n")
            f.write("\n")


def main() -> None:
    """Run merge pipeline."""
    parser = argparse.ArgumentParser(description="Merge entropy with outcomes")
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
    print(f"Loaded {len(df)} sessions\n")
    
    # Merge
    merged_df, stats = merge_outcomes_with_entropy(df)
    
    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save merged CSV
    output_path = TABLES_DIR / "entropy_with_outcomes.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data: {output_path}")
    
    # Save report
    report_path = LOGS_DIR / "outcome_merge_report.txt"
    write_merge_report(stats, report_path)
    print(f"Saved merge report: {report_path}")
    print()
    
    # Console summary
    print("=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Entropy sessions:   {stats['total_entropy_sessions']}")
    print(f"Outcome sessions:   {stats['total_outcome_sessions']}")
    print(f"Matched:            {stats['matched']}")
    print(f"Match rate:         {100 * stats['matched'] / stats['total_entropy_sessions']:.1f}%")
    print()
    
    # Show outcome distribution
    with_outcomes = merged_df[merged_df["funded_rate"].notna()]
    if len(with_outcomes) > 0:
        print("OUTCOME DISTRIBUTION:")
        print(f"  Sessions with any funded team: {with_outcomes['any_funded'].sum()}")
        print(f"  Mean funding rate:             {with_outcomes['funded_rate'].mean():.2f}")
        print(f"  Median funding rate:           {with_outcomes['funded_rate'].median():.2f}")
        print()
        print("  Funded rate distribution:")
        print(with_outcomes["funded_rate"].value_counts().sort_index().to_string())
    else:
        print("⚠️ No sessions have outcome data")


if __name__ == "__main__":
    main()
