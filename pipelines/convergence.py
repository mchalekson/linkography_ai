#!/usr/bin/env python
"""Slide 2: Convergence detection

make_slide2_convergence.py

Replicate Slide 10 - Part 2: Convergence Detection and Visualization

Detects "strict convergence" as:
  (convergence phrase) AND (commitment code) AND NOT (structural wrap text)

Computes per-60s-bin aggregations of strict convergence minutes and structural
wrap minutes, smooths them, and plots on a single axis with legend labels:
  - "Coordination/Decision (minutes)"
  - "Structural Closure (meeting management)"

Extracts the longest strict convergence utterance as a callout for speaker notes.

Usage:
  python pipelines/make_slide2_convergence.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
  python pipelines/make_slide2_convergence.py --session ... --bin-sec 60 --smooth-window 3 --out-fig figures/generated/slide2_custom.png
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONSTANTS & PATTERNS
# ============================================================

COMMITMENT_CODES = {"Coordination and Decision Practices"}

STRUCTURAL_WRAP_PAT = re.compile(
    r"\b("
    r"wrap up|time|times up|run out of time|hard stop|"
    r"next meeting|next steps|follow up|send|email|"
    r"slides?|deck|presentation|present|report out|report-out|"
    r"agenda|minutes|summar(y|ize)|"
    r"let's move on|we should stop|have to go|"
    r"screen sharing|screenshare|slide number"
    r")\b",
    flags=re.IGNORECASE
)

CONVERGENCE_PAT = re.compile(
    r"\b("
    r"we (all )?agree|consensus|settle on|go with|we'll go with|"
    r"we decide|we decided|final decision|the plan is|"
    r"we will do|we're going to do"
    r")\b",
    flags=re.IGNORECASE
)


# ============================================================
# UTILITIES
# ============================================================

def time_str_to_sec(s: str) -> float:
    """Parse 'MM:SS' or 'HH:MM:SS' string to seconds."""
    if not isinstance(s, str) or ":" not in s:
        return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return np.nan
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return np.nan


def extract_utterances_scialog(session_json: Dict) -> pd.DataFrame:
    """
    Extract utterances from SCIALOG session JSON.

    Expected schema:
      session_json["all_data"] is a list of utterance dicts with:
        - start_time, end_time (MM:SS or HH:MM:SS)
        - transcript (text)
        - annotations (dict where keys are code names)

    Returns:
      DataFrame with columns:
        idx, start_time, end_time, start_sec, end_sec, text, codes,
        dur_sec, t_sec, t_min
    """
    rows = []
    for i, u in enumerate(session_json.get("all_data", [])):
        if not isinstance(u, dict):
            continue

        start_time = u.get("start_time")
        end_time = u.get("end_time")
        start_sec = time_str_to_sec(start_time)
        end_sec = time_str_to_sec(end_time)

        text = (u.get("transcript", "") or "").strip()
        ann = u.get("annotations", {})
        codes = list(ann.keys()) if isinstance(ann, dict) else []

        rows.append({
            "idx": i,
            "start_time": start_time,
            "end_time": end_time,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text,
            "codes": codes,
        })

    df = pd.DataFrame(rows)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["start_sec"]).sort_values("start_sec").reset_index(drop=True)

    # Backfill missing end times
    df["next_start_sec"] = df["start_sec"].shift(-1)
    df["end_sec"] = df["end_sec"].fillna(df["next_start_sec"])
    df["end_sec"] = df["end_sec"].fillna(df["start_sec"])

    df["dur_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=0)
    df["t_sec"] = df["start_sec"]
    df["t_min"] = df["t_sec"] / 60.0

    return df.drop(columns=["next_start_sec"])


def has_any_code(codes, target_set: set) -> bool:
    """Check if any code in codes list is in target_set."""
    if not isinstance(codes, list):
        return False
    return any(c in target_set for c in codes)


def clean_text(s: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", str(s)).strip()


# ============================================================
# MAIN PIPELINE
# ============================================================

def compute_slide2_convergence(
    session_fp: Path,
    bin_sec: int = 60,
    smooth_window: int = 3,
    last_third_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load session, detect signals, bin, and smooth.

    Returns:
      (df, bin_summary, plot_df)
        df: full utterance-level dataframe with signal columns
        bin_summary: per-bin aggregations and smoothed signals
        plot_df: utterances in the plotted window (e.g., last third)
    """
    with open(session_fp, "r") as f:
        session = json.load(f)

    # Extract utterances
    df = extract_utterances_scialog(session)

    # Compute signal flags
    df["is_commitment_code"] = df["codes"].apply(lambda cs: has_any_code(cs, COMMITMENT_CODES))
    df["is_structural_wrap_text"] = df["text"].apply(lambda t: bool(STRUCTURAL_WRAP_PAT.search(t)))
    df["is_convergence_phrase"] = df["text"].apply(lambda t: bool(CONVERGENCE_PAT.search(t)))

    # Strict convergence = (agreement phrase) AND (commitment code) AND NOT (structural wrap)
    df["is_strict_convergence"] = (
        df["is_convergence_phrase"]
        & df["is_commitment_code"]
        & (~df["is_structural_wrap_text"])
    )

    # Optional: restrict to last third of meeting
    if last_third_only:
        meeting_start = float(df["start_sec"].min())
        meeting_end = float(df["end_sec"].max())
        meeting_len = max(1.0, meeting_end - meeting_start)
        last_third_start = meeting_start + 2.0 * meeting_len / 3.0
        plot_df = df[df["t_sec"] >= last_third_start].copy()
    else:
        plot_df = df.copy()

    # Bin and aggregate
    plot_df["t_bin"] = (plot_df["t_sec"] // bin_sec).astype(int) * bin_sec

    bin_summary = (
        plot_df.groupby("t_bin")
        .agg(
            strict_conv_time_sec=(
                "dur_sec",
                lambda s: float(s[plot_df.loc[s.index, "is_strict_convergence"]].sum()),
            ),
            structural_time_sec=(
                "dur_sec",
                lambda s: float(s[plot_df.loc[s.index, "is_structural_wrap_text"]].sum()),
            ),
        )
        .reset_index()
    )

    bin_summary["t_min"] = bin_summary["t_bin"] / 60.0
    bin_summary["strict_conv_min"] = bin_summary["strict_conv_time_sec"] / 60.0
    bin_summary["structural_min"] = bin_summary["structural_time_sec"] / 60.0

    # Smooth
    bin_summary["strict_conv_smooth"] = bin_summary["strict_conv_min"].rolling(smooth_window, 1).mean()
    bin_summary["structural_smooth"] = bin_summary["structural_min"].rolling(smooth_window, 1).mean()

    return df, bin_summary, plot_df


def plot_slide2(bin_summary: pd.DataFrame, session_stem: str, bin_sec: int = 60, last_third_only: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create the Slide 2 plot with two lines on a single axis.

    Returns:
      (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        bin_summary["t_min"],
        bin_summary["strict_conv_smooth"],
        label="Coordination/Decision (minutes)",
        linewidth=2,
    )
    ax.plot(
        bin_summary["t_min"],
        bin_summary["structural_smooth"],
        label="Structural Closure (meeting management)",
        linewidth=2,
    )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(f"Minutes of meeting time per {bin_sec}s window (smoothed)")

    title_suffix = " (last third)" if last_third_only else ""
    ax.set_title(f"What convergence looks like in practice{title_suffix}\n{session_stem}")

    ax.legend()
    fig.tight_layout()

    return fig, ax


def extract_callout(plot_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Extract the longest strict convergence utterance.

    Returns:
      (found: bool, callout_text: str)
    """
    true_conv = plot_df[plot_df["is_strict_convergence"]].copy()
    true_conv["len"] = true_conv["text"].astype(str).str.len()

    if len(true_conv) == 0:
        return False, "No strict convergence utterances in the plotted window."

    best = true_conv.sort_values("len", ascending=False).iloc[0]
    full = clean_text(best["text"])
    callout = f'({best["start_time"]}–{best["end_time"]}) "{full}"'

    return True, callout


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Slide 10 Part 2: Convergence Detection and Visualization",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Path to session JSON (e.g., data/2021NES/session_data/2021_11_04_NES_S6.json)",
    )
    parser.add_argument(
        "--bin-sec",
        type=int,
        default=60,
        help="Bin width in seconds (default 60)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Smoothing window (rolling mean, default 3)",
    )
    parser.add_argument(
        "--last-third-only",
        action="store_true",
        default=True,
        help="Restrict analysis to last third of meeting (default True)",
    )
    parser.add_argument(
        "--full-duration",
        action="store_true",
        help="Override: use full meeting duration (default False)",
    )
    parser.add_argument(
        "--print-full-callout",
        action="store_true",
        default=True,
        help="Print full callout text without truncation (default True)",
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        default=False,
        help="Print context utterances around callout (default False)",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=2,
        help="Number of utterances before callout to show (default 2)",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=2,
        help="Number of utterances after callout to show (default 2)",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=None,
        help="Path to save figure (default figures/generated/slide2_<session>.png)",
    )
    parser.add_argument(
        "--out-log",
        type=str,
        default=None,
        help="Path to save log with callout (default outputs/logs/slide2_<session>.txt)",
    )

    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    session_fp = repo_root / args.session
    last_third = args.last_third_only and not args.full_duration

    if not session_fp.exists():
        print(f"ERROR: Session file not found: {session_fp}")
        sys.exit(1)

    # Default output paths
    session_stem = session_fp.stem
    if args.out_fig is None:
        fig_dir = repo_root / "figures" / "generated"
        fig_dir.mkdir(parents=True, exist_ok=True)
        args.out_fig = str(fig_dir / f"slide2_{session_stem}.png")

    if args.out_log is None:
        log_dir = repo_root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        args.out_log = str(log_dir / f"slide2_{session_stem}.txt")

    # Run analysis
    print(f"Loading session: {session_fp}")
    df, bin_summary, plot_df = compute_slide2_convergence(
        session_fp,
        bin_sec=args.bin_sec,
        smooth_window=args.smooth_window,
        last_third_only=last_third,
    )

    # Plot
    fig, ax = plot_slide2(
        bin_summary,
        session_stem,
        bin_sec=args.bin_sec,
        last_third_only=last_third,
    )
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {args.out_fig}")
    plt.close(fig)

    # Extract callout
    found, callout_text = extract_callout(plot_df)

    # Print to stdout
    print(f"\nSLIDE CALLOUT (strict convergence):")
    print(callout_text)

    # Print context if requested
    if args.print_context and found:
        # Find the utterance and print context
        true_conv = plot_df[plot_df["is_strict_convergence"]].copy()
        true_conv["len"] = true_conv["text"].astype(str).str.len()
        best = true_conv.sort_values("len", ascending=False).iloc[0]
        best_idx = int(best["idx"])

        ctx = df[
            (df["idx"] >= best_idx - args.context_before)
            & (df["idx"] <= best_idx + args.context_after)
        ].copy()

        print("\nContext window around callout:")
        for _, r in ctx.iterrows():
            tag = []
            if r.get("is_commitment_code"): tag.append("Coord/Decision")
            if r.get("is_convergence_phrase"): tag.append("Conv phrase")
            if r.get("is_structural_wrap_text"): tag.append("Structural")
            if r.get("is_strict_convergence"): tag.append("STRICT")
            tag_str = f"[{', '.join(tag)}]" if tag else ""
            print(f'- ({r["start_time"]}–{r["end_time"]}) {tag_str} {clean_text(r["text"])}')

    # Write log
    with open(args.out_log, "w") as f:
        f.write(f"Slide 10 - Part 2: Convergence Detection\n")
        f.write(f"Session: {session_stem}\n")
        f.write(f"File: {session_fp}\n")
        f.write(f"Last third only: {last_third}\n")
        f.write(f"Bin width (sec): {args.bin_sec}\n")
        f.write(f"Smooth window: {args.smooth_window}\n")
        f.write(f"\nStrictly convergent utterances found: {len(plot_df[plot_df['is_strict_convergence']])}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CALLOUT:\n")
        f.write(f"{'='*60}\n")
        f.write(callout_text + "\n")

    print(f"\nSaved log: {args.out_log}")

    # Print stats
    n_strict = len(plot_df[plot_df["is_strict_convergence"]])
    n_structural = len(plot_df[plot_df["is_structural_wrap_text"]])
    print(f"\nStats (plotted window):")
    print(f"  Strict convergence utterances: {n_strict}")
    print(f"  Structural wrap utterances: {n_structural}")


if __name__ == "__main__":
    main()
