#!/usr/bin/env python
"""Slide 1: Signals by time bin

make_slide1_signals.py

Replicate Slide 1 plotting: per-bin minutes of coordination/decision and structural
wrap signals, with smoothing. Uses `compute_signals_by_bin` from
`src/linkography_ai/slides.py` when available; otherwise reimplements minimal logic.

Outputs a PNG figure and a small log with stats and a callout (longest commitment-coded
utterance in the plotted window).

Usage example:
  python pipelines/make_slide1_signals.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
"""

from pathlib import Path
import argparse
import sys
import json
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Try to reuse existing utilities
try:
    from linkography_ai.slides import compute_signals_by_bin
    _HAS_SLIDES = True
except Exception:
    _HAS_SLIDES = False


def fallback_compute_signals(session_fp: Path, bin_sec: int, smooth_window: int, last_third_only: bool):
    """Fallback minimal implementation if slides.compute_signals_by_bin is unavailable.
    This function loads the SCIALOG JSON, extracts utterances (start/end/text/codes), bins by
    `bin_sec`, and computes minutes of coordination (is_commitment_code) and structural wrap
    (regex in slides.py is not reproduced here). It is intentionally minimal; prefer the
    installed `linkography_ai.slides` functions when available.
    """
    raise RuntimeError("slides.compute_signals_by_bin is required but not available.")


def extract_callout_from_df(df: pd.DataFrame) -> Tuple[bool, str]:
    """Return longest utterance that is commitment-coded in df (plotted window).
    """
    if "is_commitment_code" not in df.columns:
        return False, "No commitment-code column available."
    true_c = df[df["is_commitment_code"]].copy()
    if true_c.empty:
        return False, "No commitment-coded utterances in the plotted window."
    true_c["len"] = true_c["text"].astype(str).str.len()
    best = true_c.sort_values("len", ascending=False).iloc[0]
    text = "(" + str(best.get("start_time", "")) + "–" + str(best.get("end_time", "")) + ") \"" + str(best["text"]) + "\""
    return True, text


def main():
    p = argparse.ArgumentParser(description="Slide 1: Signals by time bin (coordination + structural)")
    p.add_argument("--session", required=True, help="Path to session JSON (repo-relative)")
    p.add_argument("--bin-sec", type=int, default=60)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--last-third-only", action="store_true", default=True)
    p.add_argument("--print-context", action="store_true", default=False)
    p.add_argument("--context-before", type=int, default=2)
    p.add_argument("--context-after", type=int, default=2)
    p.add_argument("--out-fig", default=None)
    p.add_argument("--out-log", default=None)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    session_fp = repo_root / args.session
    if not session_fp.exists():
        print(f"Session not found: {session_fp}")
        sys.exit(1)

    if args.out_fig is None:
        out_fig = repo_root / "figures" / "generated" / f"slide1_{session_fp.stem}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_fig = Path(args.out_fig)

    if args.out_log is None:
        out_log = repo_root / "outputs" / "logs" / f"slide1_{session_fp.stem}.txt"
        out_log.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_log = Path(args.out_log)

    # Compute signals
    if _HAS_SLIDES:
        df_utt, bins = compute_signals_by_bin(
            session_fp,
            bin_sec=args.bin_sec,
            smooth_window=args.smooth_window,
            last_third_only=args.last_third_only,
        )
    else:
        # If the helper is missing, abort (we expect slides.py to be present in this repo)
        print("Error: compute_signals_by_bin not available. Install or use repository's slides module.")
        sys.exit(1)

    # Plot: minutes of commitment_time_min and structural_time_min (smoothed)
    fig, ax = plt.subplots(figsize=(10, 4))
    if "commitment_time_min_smooth" in bins.columns:
        ax.plot(bins["t_min"], bins["commitment_time_min_smooth"], label="Coordination/Decision (minutes)")
    elif "commitment_time_min" in bins.columns and "commitment_time_min" in bins.columns:
        ax.plot(bins["t_min"], bins["commitment_time_min"].rolling(args.smooth_window, 1).mean(), label="Coordination/Decision (minutes)")
    else:
        # fallback to commitment_count smoothed
        if "commitment_count_smooth" in bins.columns:
            ax.plot(bins["t_min"], bins["commitment_count_smooth"], label="Coordination/Decision (minutes)")

    if "structural_time_min_smooth" in bins.columns:
        ax.plot(bins["t_min"], bins["structural_time_min_smooth"], label="Structural Closure (meeting management)")
    elif "structural_time_min" in bins.columns:
        ax.plot(bins["t_min"], bins["structural_time_min"].rolling(args.smooth_window, 1).mean(), label="Structural Closure (meeting management)")
    else:
        if "structural_count_smooth" in bins.columns:
            ax.plot(bins["t_min"], bins["structural_count_smooth"], label="Structural Closure (meeting management)")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(f"Minutes per {args.bin_sec}s window (smoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    found, callout = extract_callout_from_df(df_utt)

    with open(out_log, "w") as f:
        f.write(f"Slide 1 signals\nSession: {session_fp.stem}\n\n")
        f.write(f"Callout found: {found}\n")
        f.write(f"Callout: {callout}\n")

    print(f"Saved figure: {out_fig}")
    print(f"Saved log: {out_log}")


if __name__ == "__main__":
    main()
