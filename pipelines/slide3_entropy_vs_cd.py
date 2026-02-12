#!/usr/bin/env python
"""
make_slide3_entropy_vs_cd.py

Replicate Slide 3 (Entropy vs Coordination/Decision) chunk using
`compute_entropy_vs_cd` and `plot_entropy_vs_cd` from `src/linkography_ai/slides.py`.

Produces a PNG figure and a log file containing a longest-commitment-code callout.

Usage example:
  python pipelines/make_slide3_entropy_vs_cd.py --session data/2021NES/session_data/2021_11_04_NES_S6.json
"""

from pathlib import Path
import argparse
import sys

try:
    from linkography_ai.slides import compute_entropy_vs_cd, plot_entropy_vs_cd
except Exception as e:
    print("Error importing slide utilities from linkography_ai.slides:", e)
    raise

import matplotlib.pyplot as plt
import pandas as pd
import re


def extract_callout(plot_df: pd.DataFrame) -> (bool, str):
    """Return the longest commitment-coded utterance in `plot_df` (plotted window)."""
    if "is_commitment_code" not in plot_df.columns:
        return False, "No commitment-code column available."
    true_c = plot_df[plot_df["is_commitment_code"]].copy()
    if true_c.empty:
        return False, "No commitment-coded utterances in the plotted window."
    true_c["len"] = true_c["text"].astype(str).str.len()
    best = true_c.sort_values("len", ascending=False).iloc[0]
    text = f'({best.get("start_time","")}-{best.get("end_time","")}) "{best["text"]}"'
    return True, text


def main():
    p = argparse.ArgumentParser(description="Slide 3: Entropy vs Coordination/Decision")
    p.add_argument("--session", required=True)
    p.add_argument("--bin-sec", type=int, default=60)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--last-third-only", action="store_true", default=True)
    p.add_argument("--exclude-structural", action="store_true", default=True,
                   help="Exclude structural wrap utterances from entropy computation")
    p.add_argument("--out-fig", default=None)
    p.add_argument("--out-log", default=None)
    p.add_argument("--print-context", action="store_true", default=False)
    p.add_argument("--context-before", type=int, default=2)
    p.add_argument("--context-after", type=int, default=2)

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    session_fp = repo_root / args.session
    if not session_fp.exists():
        print(f"Session not found: {session_fp}")
        sys.exit(1)

    session_stem = session_fp.stem
    if args.out_fig is None:
        out_fig = repo_root / "figures" / "generated" / f"slide3_{session_stem}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_fig = Path(args.out_fig)

    if args.out_log is None:
        out_log = repo_root / "outputs" / "logs" / f"slide3_{session_stem}.txt"
        out_log.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_log = Path(args.out_log)

    # Compute
    bins, plot_df = compute_entropy_vs_cd(
        session_fp,
        bin_sec=args.bin_sec,
        smooth_window=args.smooth_window,
        last_third_only=args.last_third_only,
        exclude_structural_from_entropy=args.exclude_structural,
    )

    # Plot
    fig, ax_entropy, ax_minutes = plot_entropy_vs_cd(
        bins,
        session_stem=session_stem,
        bin_sec=args.bin_sec,
        plot_structural=False,
    )
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Callout extraction
    found, callout = extract_callout(plot_df)

    with open(out_log, "w") as f:
        f.write(f"Slide 3: Entropy vs Coordination/Decision\n")
        f.write(f"Session: {session_stem}\n")
        f.write(f"File: {session_fp}\n")
        f.write(f"Bins (sec): {args.bin_sec}\n")
        f.write(f"Smooth window: {args.smooth_window}\n")
        f.write(f"Exclude structural from entropy: {args.exclude_structural}\n\n")
        f.write(f"Callout found: {found}\n")
        f.write(callout + "\n")

    print(f"Saved figure: {out_fig}")
    print(f"Saved log: {out_log}")


if __name__ == '__main__':
    main()
