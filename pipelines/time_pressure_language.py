#!/usr/bin/env python
"""Time-pressure and decision-closure language analysis.

Scans utterances for time-pressure or decision-closure phrases and summarizes
counts overall and by time-based thirds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from linkography_ai.discovery import list_conferences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"

TIME_PRESSURE_PAT = re.compile(
    r"\b("
    r"time( is|')? (up|short|limited)|times up|run out of time|hard stop|"
    r"minutes left|only \d+ minutes|we have \d+ minutes|"
    r"need to wrap up|wrap up|before we end|"
    r"deadline|by the end|end of the session|"
    r"we should decide|need to decide|make a decision|"
    r"let's decide|finalize|lock in"
    r")\b",
    flags=re.IGNORECASE,
)

DECISION_CLOSURE_PAT = re.compile(
    r"\b("
    r"final decision|we decided|decision is|we'll go with|we will do|"
    r"go with|settle on|consensus|agree to|"
    r"the plan is|we are going to|we're going to"
    r")\b",
    flags=re.IGNORECASE,
)


def time_str_to_sec(s: Any) -> float:
    if not isinstance(s, str) or ":" not in s:
        return math.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return math.nan
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return math.nan


def assign_time_third(t_sec: float, start: float, end: float) -> str:
    total = max(1.0, end - start)
    one_third = start + total / 3.0
    two_third = start + 2.0 * total / 3.0
    if t_sec < one_third:
        return "beginning"
    if t_sec < two_third:
        return "middle"
    return "end"


def extract_cdp_flag(u: Dict[str, Any]) -> bool:
    ad = u.get("annotation_dict") or u.get("annotations") or {}
    if isinstance(ad, dict):
        return "Coordination and Decision Practices" in ad
    return False


def load_session(session_path: Path) -> list[dict]:
    obj = json.loads(session_path.read_text())
    if isinstance(obj, dict) and isinstance(obj.get("all_data"), list):
        return obj["all_data"]
    if isinstance(obj, dict) and isinstance(obj.get("utterances"), list):
        return obj["utterances"]
    if isinstance(obj, list):
        return obj
    return []


def analyze_session(session_path: Path) -> dict | None:
    utter_list = load_session(session_path)
    if not utter_list:
        return None

    times = [time_str_to_sec(u.get("start_time")) for u in utter_list]
    times = [t for t in times if not math.isnan(t)]
    if not times:
        return None

    t_start, t_end = min(times), max(times)
    row = {
        "session_id": session_path.stem,
        "n_utterances": len(utter_list),
        "time_pressure_total": 0,
        "decision_closure_total": 0,
        "time_pressure_cdp": 0,
        "decision_closure_cdp": 0,
    }

    for seg in ["beginning", "middle", "end"]:
        row[f"time_pressure_{seg}"] = 0
        row[f"decision_closure_{seg}"] = 0

    for u in utter_list:
        text = (u.get("transcript") or "").strip()
        start_sec = time_str_to_sec(u.get("start_time"))
        if math.isnan(start_sec):
            continue
        seg = assign_time_third(start_sec, t_start, t_end)
        is_time = bool(TIME_PRESSURE_PAT.search(text))
        is_close = bool(DECISION_CLOSURE_PAT.search(text))
        is_cdp = extract_cdp_flag(u)

        if is_time:
            row["time_pressure_total"] += 1
            row[f"time_pressure_{seg}"] += 1
            if is_cdp:
                row["time_pressure_cdp"] += 1
        if is_close:
            row["decision_closure_total"] += 1
            row[f"decision_closure_{seg}"] += 1
            if is_cdp:
                row["decision_closure_cdp"] += 1

    return row


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("TIME PRESSURE LANGUAGE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(df)}\n\n")

        for metric in ["time_pressure", "decision_closure"]:
            f.write(f"{metric.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total mean: {df[f'{metric}_total'].mean():.2f}\n")
            f.write(f"Total median: {df[f'{metric}_total'].median():.2f}\n")
            f.write("By segment (mean counts):\n")
            for seg in ["beginning", "middle", "end"]:
                f.write(f"  {seg}: {df[f'{metric}_{seg}'].mean():.2f}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze time-pressure language")
    parser.add_argument("--conference", default="ALL", help="Conference code or ALL")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences() if args.conference.upper() == "ALL" else [args.conference]
    rows = []

    for conf in conferences:
        session_dir = DATA_DIR / conf / "session_data"
        if not session_dir.exists():
            continue
        for session_path in sorted(session_dir.glob("*.json")):
            row = analyze_session(session_path)
            if row is None:
                continue
            row["conference"] = conf
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "time_pressure_language_by_session.csv"
    df.to_csv(out_path, index=False)

    summary_path = ANALYSIS_DIR / "time_pressure_language_summary.txt"
    write_summary(df, summary_path)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
