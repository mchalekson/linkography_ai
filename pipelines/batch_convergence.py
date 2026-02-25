#!/usr/bin/env python
"""Batch Convergence Detection

Computes strict convergence and structural wrap rates across all sessions.
Produces per-session table and optional convergence-vs-entropy scatter plot.

Usage:
    python pipelines/batch_convergence.py
    python pipelines/batch_convergence.py --conference ALL
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linkography_ai.discovery import list_conferences
from linkography_ai.slides import DEFAULT_COMMITMENT_CODES, DEFAULT_STRUCTURAL_WRAP_PAT

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
LOGS_DIR = OUT_DIR / "logs"
FIGURES_DIR = REPO_ROOT / "figures" / "final"

CONVERGENCE_PAT = re.compile(
    r"\b("
    r"we (all )?agree|consensus|settle on|go with|we'll go with|"
    r"we decide|we decided|final decision|the plan is|"
    r"we will do|we're going to do"
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


def load_session_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_utterances(session_json: dict) -> pd.DataFrame:
    rows = []
    utter_list = session_json.get("all_data")
    if not isinstance(utter_list, list):
        return pd.DataFrame(columns=[
            "idx", "start_sec", "end_sec", "text", "codes",
        ])

    for i, u in enumerate(utter_list):
        if not isinstance(u, dict):
            continue
        start_time = u.get("start_time")
        end_time = u.get("end_time")
        start_sec = time_str_to_sec(start_time)
        end_sec = time_str_to_sec(end_time)
        text = (u.get("transcript") or "").strip()
        ann = u.get("annotations", {})
        codes = list(ann.keys()) if isinstance(ann, dict) else []
        rows.append({
            "idx": i,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text,
            "codes": codes,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["start_sec"]).sort_values("start_sec").reset_index(drop=True)

    df["next_start_sec"] = df["start_sec"].shift(-1)
    df["end_sec"] = df["end_sec"].fillna(df["next_start_sec"]).fillna(df["start_sec"])
    df["end_sec"] = np.maximum(df["end_sec"].to_numpy(), df["start_sec"].to_numpy())
    df["dur_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=0)

    return df.drop(columns=["next_start_sec"])


def compute_session_metrics(session_path: Path) -> Optional[dict]:
    session = load_session_json(session_path)
    df = extract_utterances(session)
    if df.empty:
        return None

    df["is_commitment_code"] = df["codes"].apply(
        lambda cs: any(c in DEFAULT_COMMITMENT_CODES for c in cs) if isinstance(cs, list) else False
    )
    df["is_structural_wrap"] = df["text"].apply(lambda t: bool(DEFAULT_STRUCTURAL_WRAP_PAT.search(str(t))))
    df["is_convergence_phrase"] = df["text"].apply(lambda t: bool(CONVERGENCE_PAT.search(str(t))))
    df["is_strict_convergence"] = (
        df["is_convergence_phrase"] & df["is_commitment_code"] & (~df["is_structural_wrap"])
    )

    total_time = float(df["dur_sec"].sum())
    if total_time <= 0:
        total_time = float(df["end_sec"].max() - df["start_sec"].min())

    meeting_start = float(df["start_sec"].min())
    meeting_end = float(df["end_sec"].max())
    meeting_len = max(1.0, meeting_end - meeting_start)
    last_third_start = meeting_start + 2.0 * meeting_len / 3.0

    last_third = df[df["start_sec"] >= last_third_start]

    strict_conv_time = float(df.loc[df["is_strict_convergence"], "dur_sec"].sum())
    structural_time = float(df.loc[df["is_structural_wrap"], "dur_sec"].sum())

    strict_conv_time_last = float(last_third.loc[last_third["is_strict_convergence"], "dur_sec"].sum())
    structural_time_last = float(last_third.loc[last_third["is_structural_wrap"], "dur_sec"].sum())
    last_third_time = float(last_third["dur_sec"].sum()) if not last_third.empty else 0.0

    n_utts = len(df)
    n_strict = int(df["is_strict_convergence"].sum())
    n_struct = int(df["is_structural_wrap"].sum())

    return {
        "session_id": session_path.stem,
        "n_utterances": n_utts,
        "strict_conv_utt": n_strict,
        "structural_wrap_utt": n_struct,
        "strict_conv_time_sec": strict_conv_time,
        "structural_wrap_time_sec": structural_time,
        "total_time_sec": total_time,
        "strict_conv_rate": strict_conv_time / total_time if total_time > 0 else math.nan,
        "structural_wrap_rate": structural_time / total_time if total_time > 0 else math.nan,
        "strict_conv_time_last_third_sec": strict_conv_time_last,
        "structural_wrap_time_last_third_sec": structural_time_last,
        "last_third_time_sec": last_third_time,
        "strict_conv_rate_last_third": strict_conv_time_last / last_third_time if last_third_time > 0 else math.nan,
        "structural_wrap_rate_last_third": structural_time_last / last_third_time if last_third_time > 0 else math.nan,
    }


def find_latest_entropy_csv() -> Optional[Path]:
    csvs = list(TABLES_DIR.glob("cdp_entropy_by_session_*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def plot_convergence_vs_entropy(convergence_df: pd.DataFrame) -> Optional[Path]:
    entropy_csv = find_latest_entropy_csv()
    if entropy_csv is None or not entropy_csv.exists():
        return None

    ent = pd.read_csv(entropy_csv)
    ent["entropy_change"] = ent["entropy_end"] - ent["entropy_beginning"]
    merged = convergence_df.merge(ent[["session_id", "entropy_change"]], on="session_id", how="inner")
    merged = merged.dropna(subset=["strict_conv_rate_last_third", "entropy_change"])

    if merged.empty:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "convergence_vs_entropy_scatter.png"

    plt.figure(figsize=(6, 4))
    plt.scatter(merged["strict_conv_rate_last_third"], merged["entropy_change"], alpha=0.6)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("Strict convergence rate (last third, time-based)")
    plt.ylabel("Entropy change (end - beginning)")
    plt.title("Convergence vs Entropy Change")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convergence detection")
    parser.add_argument("--conference", default="ALL", help="Conference code or ALL")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences() if args.conference.upper() == "ALL" else [args.conference]
    rows = []

    for conf in conferences:
        session_dir = DATA_DIR / conf / "session_data"
        if not session_dir.exists():
            continue
        for session_path in sorted(session_dir.glob("*.json")):
            metrics = compute_session_metrics(session_path)
            if metrics is None:
                continue
            metrics["conference"] = conf
            rows.append(metrics)

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "convergence_rates_by_session.csv"
    df.to_csv(out_path, index=False)

    fig_path = plot_convergence_vs_entropy(df)

    log_path = LOGS_DIR / "batch_convergence_report.txt"
    with open(log_path, "w") as f:
        f.write("Batch Convergence Detection\n")
        f.write(f"sessions_total={len(df)}\n")
        f.write(f"output_table={out_path}\n")
        if fig_path:
            f.write(f"scatter_figure={fig_path}\n")

    print(f"Saved: {out_path}")
    if fig_path:
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
