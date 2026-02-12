#!/usr/bin/env python
"""Data Validation Pipeline

Validates integrity of session JSON files across all conferences.
Checks for required fields, missing timestamps, and CDP annotation coverage.

Usage:
    python pipelines/validate_data_integrity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
LOGS_DIR = OUT_DIR / "logs"


def discover_conferences() -> List[str]:
    """Find all conference directories with session_data subdirs."""
    conferences = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "session_data").exists():
            conferences.append(p.name)
    return conferences


def validate_session(session_path: Path) -> Dict[str, any]:
    """Validate a single session JSON file.
    
    Returns:
        dict with validation results
    """
    results = {
        "path": str(session_path.relative_to(REPO_ROOT)),
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        data = json.loads(session_path.read_text())
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"JSON parse error: {e}")
        return results
    
    # Check all_data exists
    if "all_data" not in data:
        results["valid"] = False
        results["errors"].append("Missing 'all_data' key")
        return results
    
    if not isinstance(data["all_data"], list):
        results["valid"] = False
        results["errors"].append("'all_data' is not a list")
        return results
    
    utterances = data["all_data"]
    n = len(utterances)
    results["stats"]["n_utterances"] = n
    
    if n == 0:
        results["warnings"].append("Empty all_data array")
        return results
    
    # Check timestamps
    missing_start = 0
    missing_end = 0
    missing_both = 0
    
    for u in utterances:
        has_start = "start_time" in u and u["start_time"]
        has_end = "end_time" in u and u["end_time"]
        
        if not has_start:
            missing_start += 1
        if not has_end:
            missing_end += 1
        if not has_start and not has_end:
            missing_both += 1
    
    results["stats"]["missing_start_time"] = missing_start
    results["stats"]["missing_end_time"] = missing_end
    results["stats"]["missing_both_times"] = missing_both
    results["stats"]["pct_missing_times"] = round(100 * missing_both / n, 1) if n > 0 else 0
    
    # Check annotations
    has_annotations = 0
    has_cdp_annotations = 0
    
    for u in utterances:
        if "annotations" in u and isinstance(u["annotations"], dict):
            has_annotations += 1
            # Check for any CDP-like key
            for key in u["annotations"]:
                if "coordination" in key.lower() or "decision" in key.lower() or "cdp" in key.lower():
                    has_cdp_annotations += 1
                    break
    
    results["stats"]["has_annotations"] = has_annotations
    results["stats"]["has_cdp_annotations"] = has_cdp_annotations
    results["stats"]["pct_with_annotations"] = round(100 * has_annotations / n, 1) if n > 0 else 0
    results["stats"]["pct_with_cdp"] = round(100 * has_cdp_annotations / n, 1) if n > 0 else 0
    
    # Warnings
    if results["stats"]["pct_missing_times"] > 10:
        results["warnings"].append(f"{results['stats']['pct_missing_times']}% utterances missing timestamps")
    
    if results["stats"]["pct_with_annotations"] < 50:
        results["warnings"].append(f"Only {results['stats']['pct_with_annotations']}% of utterances have annotations")
    
    return results


def main() -> None:
    """Run validation across all conferences."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Validating data integrity...")
    print(f"Data directory: {DATA_DIR}")
    print()
    
    conferences = discover_conferences()
    print(f"Found {len(conferences)} conferences: {', '.join(conferences)}")
    print()
    
    all_results = []
    total_sessions = 0
    total_valid = 0
    total_with_errors = 0
    total_with_warnings = 0
    
    for conf in conferences:
        session_dir = DATA_DIR / conf / "session_data"
        session_files = sorted(session_dir.glob("*.json"))
        
        for sf in session_files:
            total_sessions += 1
            result = validate_session(sf)
            all_results.append(result)
            
            if result["valid"]:
                total_valid += 1
            if result["errors"]:
                total_with_errors += 1
            if result["warnings"]:
                total_with_warnings += 1
    
    # Write detailed report
    report_path = LOGS_DIR / "data_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total sessions: {total_sessions}\n")
        f.write(f"Valid sessions: {total_valid}\n")
        f.write(f"Sessions with errors: {total_with_errors}\n")
        f.write(f"Sessions with warnings: {total_with_warnings}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        for r in all_results:
            f.write(f"Session: {r['path']}\n")
            f.write(f"  Valid: {r['valid']}\n")
            
            if r["errors"]:
                f.write(f"  Errors:\n")
                for err in r["errors"]:
                    f.write(f"    - {err}\n")
            
            if r["warnings"]:
                f.write(f"  Warnings:\n")
                for warn in r["warnings"]:
                    f.write(f"    - {warn}\n")
            
            if r["stats"]:
                f.write(f"  Stats:\n")
                for k, v in r["stats"].items():
                    f.write(f"    {k}: {v}\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("-" * 80 + "\n")
        f.write("AGGREGATE STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        total_utterances = sum(r["stats"].get("n_utterances", 0) for r in all_results if r["stats"])
        total_with_cdp = sum(r["stats"].get("has_cdp_annotations", 0) for r in all_results if r["stats"])
        
        f.write(f"Total utterances across all sessions: {total_utterances}\n")
        f.write(f"Utterances with CDP annotations: {total_with_cdp}\n")
        
        if total_utterances > 0:
            pct_cdp = round(100 * total_with_cdp / total_utterances, 1)
            f.write(f"Overall CDP annotation coverage: {pct_cdp}%\n")
    
    # Console summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total sessions:         {total_sessions}")
    print(f"Valid sessions:         {total_valid}")
    print(f"Sessions with errors:   {total_with_errors}")
    print(f"Sessions with warnings: {total_with_warnings}")
    print()
    print(f"Detailed report: {report_path}")
    print()
    
    if total_with_errors > 0:
        print("⚠️  Some sessions have validation errors. See report for details.")
    elif total_with_warnings > 0:
        print("⚠️  Some sessions have warnings. See report for details.")
    else:
        print("✅ All sessions passed validation!")


if __name__ == "__main__":
    main()
