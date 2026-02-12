# src/linkography_ai/slides.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Default signal definitions
# ----------------------------

DEFAULT_COMMITMENT_CODES = {
    "Coordination and Decision Practices",
    "Coordination/Decision Practices",
    "Coordination & Decision Practices",
    "Decision Practices",
    "Coordination Practices",
    "Commitment/Closure",
}

DEFAULT_STRUCTURAL_WRAP_PAT = re.compile(
    r"\b("
    r"wrap up|time|times up|run out of time|hard stop|"
    r"next meeting|next steps|follow up|send|email|"
    r"slides?|deck|presentation|present|report out|report-out|"
    r"agenda|minutes|summar(y|ize)|"
    r"let's move on|we should stop|have to go|"
    r"zoom|screen share|screenshare|screen sharing|slide number|main room|breakout"
    r")\b",
    flags=re.IGNORECASE,
)

# Tiny stopword set: keeps entropy lightweight and dependency-free
DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "in", "on", "for", "with",
    "we", "i", "you", "they", "he", "she", "it", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may", "might",
    "at", "as", "by", "from", "about", "into", "over", "under", "than", "then",
    "yeah", "okay", "ok", "um", "uh", "like",
}

TOKEN_PAT = re.compile(r"[a-zA-Z']+")


# ----------------------------
# Low-level utilities
# ----------------------------

def _time_str_to_sec(s: Any) -> float:
    """Parse 'MM:SS' or 'HH:MM:SS' -> seconds."""
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


def _unwrap_monotonic(
    sec_series: pd.Series,
    modulus: int = 3600,
    tol: float = 1.0,
    reset_high_frac: float = 0.80,   # prev near end of hour
    reset_low_frac: float = 0.20,    # current near start of hour
) -> pd.Series:
    """
    Make times monotonic.

    Only treat a backwards jump as a TRUE reset if:
      - previous time is near the end of the modulus window (e.g., > 0.8 * 3600)
      - current time is near the beginning (e.g., < 0.2 * 3600)

    Otherwise, if we see small/moderate backwards movement (out-of-order rows, jitter),
    we DO NOT add +3600. We clamp to keep monotonic.
    """
    secs = pd.to_numeric(sec_series, errors="coerce").astype(float).to_numpy()
    out = np.full_like(secs, np.nan, dtype=float)

    offset = 0.0
    prev = None

    hi = modulus * reset_high_frac
    lo = modulus * reset_low_frac

    for i, s in enumerate(secs):
        if np.isnan(s):
            continue

        cur = s + offset

        if prev is not None and cur < prev - tol:
            # Check if this looks like a genuine timer reset (e.g., 59:xx -> 00:yy)
            prev_mod = prev % modulus
            s_mod = s % modulus

            is_true_reset = (prev_mod >= hi) and (s_mod <= lo)

            if is_true_reset:
                offset += modulus
                cur = s + offset
            else:
                # Not a reset: it's out-of-order data or tiny jitter.
                # Clamp so time never goes backwards.
                cur = prev

        out[i] = cur
        prev = cur

    return pd.Series(out, index=sec_series.index)


def _has_any_code(codes: Any, target_set: set[str]) -> bool:
    if not isinstance(codes, list):
        return False
    return any(c in target_set for c in codes)


def _clean_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    toks = [t.lower() for t in TOKEN_PAT.findall(text or "")]
    toks = [t for t in toks if t not in stopwords and len(t) > 2]
    return toks


def _shannon_entropy(tokens: list[str]) -> float:
    if not tokens:
        return np.nan
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _load_session_json(session_fp: Path) -> dict:
    with open(session_fp, "r") as f:
        return json.load(f)


def _extract_utterances_scialog(session_json: dict) -> pd.DataFrame:
    """
    SCIALOG-ish schema:
      session_json["all_data"] is a list of dicts with:
        - start_time, end_time (MM:SS or HH:MM:SS)
        - transcript (text)
        - annotations (dict of code -> metadata dict)
    Returns canonical utterance df with:
      idx, start_time, end_time, start_sec_raw, end_sec_raw, start_sec, end_sec,
      dur_sec, t_sec, t_min, text, codes
    """
    utt_list = session_json.get("all_data")
    if not isinstance(utt_list, list):
        raise ValueError("Expected session_json['all_data'] to be a list (SCIALOG schema).")

    rows: list[dict] = []
    for i, u in enumerate(utt_list):
        if not isinstance(u, dict):
            continue

        start_time = u.get("start_time")
        end_time = u.get("end_time")

        start_sec_raw = _time_str_to_sec(start_time)
        end_sec_raw = _time_str_to_sec(end_time)

        text = (u.get("transcript", "") or "").strip()

        ann = u.get("annotations", {})
        if isinstance(ann, dict):
            codes = list(ann.keys())
        elif isinstance(ann, list):
            # just in case something weird got saved
            codes = [str(x) for x in ann]
        else:
            codes = []

        rows.append(
            {
                "idx": i,
                "start_time": start_time,
                "end_time": end_time,
                "start_sec_raw": start_sec_raw,
                "end_sec_raw": end_sec_raw,
                "text": text,
                "codes": codes,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No utterances extracted from session_json['all_data'].")

    df["start_sec_raw"] = pd.to_numeric(df["start_sec_raw"], errors="coerce")
    df["end_sec_raw"] = pd.to_numeric(df["end_sec_raw"], errors="coerce")

    df = df.dropna(subset=["start_sec_raw"]).reset_index(drop=True)

    # IMPORTANT: Some session JSONs are not strictly ordered.
    # Sort by raw start time BEFORE unwrapping so we don't interpret out-of-order rows as "resets".
    df = df.sort_values(["start_sec_raw", "end_sec_raw"], kind="mergesort").reset_index(drop=True)

    # monotonic unwrap (handles resets)
    df["start_sec"] = _unwrap_monotonic(df["start_sec_raw"], modulus=3600, tol=1.0)
    df["end_sec"] = _unwrap_monotonic(df["end_sec_raw"], modulus=3600, tol=1.0)

    df["end_sec"] = np.maximum(df["end_sec"].to_numpy(), df["start_sec"].to_numpy())

    # fill missing ends with next start
    df["next_start_sec"] = df["start_sec"].shift(-1)
    df["end_sec"] = df["end_sec"].fillna(df["next_start_sec"])
    df["end_sec"] = df["end_sec"].fillna(df["start_sec"])

    df["dur_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=0)
    df["t_sec"] = df["start_sec"]
    df["t_min"] = df["t_sec"] / 60.0

    df["text"] = df["text"].fillna("").astype(str)
    df["codes"] = df["codes"].apply(lambda x: x if isinstance(x, list) else [])

    df = df.sort_values("t_sec").reset_index(drop=True)
    return df.drop(columns=["next_start_sec"])


def _restrict_last_third(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    meeting_start = float(df["start_sec"].min())
    meeting_end = float(df["end_sec"].max())
    meeting_len = max(1.0, meeting_end - meeting_start)
    last_third_start = meeting_start + 2.0 * meeting_len / 3.0
    return df[df["t_sec"] >= last_third_start].copy()


def _add_bins(df: pd.DataFrame, bin_sec: int) -> pd.DataFrame:
    df = df.copy()
    df["t_bin"] = (df["t_sec"] // bin_sec).astype(int) * bin_sec
    return df


def _smooth_cols(bins: pd.DataFrame, cols: Iterable[str], smooth_window: int) -> pd.DataFrame:
    bins = bins.copy()
    for col in cols:
        bins[f"{col}_smooth"] = bins[col].rolling(smooth_window, min_periods=1).mean()
    return bins


# ============================================================
# 1) Slide 1/2 style: signals by time bin
# ============================================================

def compute_signals_by_bin(
    session_fp: Path | str,
    *,
    bin_sec: int = 60,
    smooth_window: int = 3,
    last_third_only: bool = False,
    commitment_codes: Optional[set[str]] = None,
    structural_wrap_pat: Optional[re.Pattern] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slide 1/2 core: compute coordination/decision signal + structural wrap-up signal
    and summarize by time bin (counts + minutes), with smoothing.

    Returns:
      df_utt: utterance-level df with signal columns + t_bin
      bins: per-bin summary (counts + minutes) + smoothed columns
    """
    session_fp = Path(session_fp)
    session_json = _load_session_json(session_fp)
    df = _extract_utterances_scialog(session_json)

    if last_third_only:
        df = _restrict_last_third(df)

    commitment_codes = commitment_codes or DEFAULT_COMMITMENT_CODES
    structural_wrap_pat = structural_wrap_pat or DEFAULT_STRUCTURAL_WRAP_PAT

    df["is_commitment_code"] = df["codes"].apply(lambda cs: _has_any_code(cs, commitment_codes))
    df["is_structural_wrap_text"] = df["text"].apply(lambda t: bool(structural_wrap_pat.search(t or "")))

    df = _add_bins(df, bin_sec=bin_sec)

    # group aggregations
    def _sum_dur_where(mask_col: str):
        def _inner(s: pd.Series) -> float:
            idxs = s.index
            mask = df.loc[idxs, mask_col]
            return float(s[mask].sum())
        return _inner

    bins = (
        df.groupby("t_bin")
          .agg(
              n_utt=("idx", "count"),
              commitment_count=("is_commitment_code", "sum"),
              structural_count=("is_structural_wrap_text", "sum"),
              commitment_time_sec=("dur_sec", _sum_dur_where("is_commitment_code")),
              structural_time_sec=("dur_sec", _sum_dur_where("is_structural_wrap_text")),
          )
          .reset_index()
          .sort_values("t_bin")
    )

    bins["t_min"] = bins["t_bin"] / 60.0
    bins["commitment_time_min"] = bins["commitment_time_sec"] / 60.0
    bins["structural_time_min"] = bins["structural_time_sec"] / 60.0

    bins = _smooth_cols(
        bins,
        cols=["commitment_count", "structural_count", "commitment_time_min", "structural_time_min"],
        smooth_window=smooth_window,
    )

    return df, bins


# ============================================================
# 2) Slide 3 style: entropy vs coordination/decision
# ============================================================

def compute_entropy_vs_cd(
    session_fp: Path | str,
    *,
    bin_sec: int = 60,
    smooth_window: int = 3,
    last_third_only: bool = True,
    exclude_structural_from_entropy: bool = True,
    commitment_codes: Optional[set[str]] = None,
    structural_wrap_pat: Optional[re.Pattern] = None,
    stopwords: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes per-bin:
      - cd_min: minutes of Coordination/Decision utterances per bin
      - entropy: Shannon entropy of token distribution per bin

    IMPORTANT: bins are indexed in MEETING-RELATIVE minutes (minutes since meeting start),
    even when last_third_only=True. This matches your original notebook plots where
    the last-third window appears around ~50–75 minutes (not 0–25).
    """
    session_fp = Path(session_fp)
    session_json = _load_session_json(session_fp)
    df_full = _extract_utterances_scialog(session_json)

    commitment_codes = commitment_codes or DEFAULT_COMMITMENT_CODES
    structural_wrap_pat = structural_wrap_pat or DEFAULT_STRUCTURAL_WRAP_PAT
    stopwords = stopwords or DEFAULT_STOPWORDS

    df_full = df_full.copy()
    df_full["is_commitment_code"] = df_full["codes"].apply(lambda cs: _has_any_code(cs, commitment_codes))
    df_full["is_structural_wrap"] = df_full["text"].apply(lambda t: bool(structural_wrap_pat.search(t or "")))

    # meeting start (absolute seconds)
    meeting_t0 = float(df_full["t_sec"].min()) if not df_full.empty else 0.0

    # restrict window
    df = _restrict_last_third(df_full) if last_third_only else df_full.copy()

    # meeting-relative time
    df["t_rel_sec"] = df["t_sec"] - meeting_t0
    df["t_rel_min"] = df["t_rel_sec"] / 60.0

    # keep BOTH bins (avoid semantic confusion later)
    df["t_bin_abs"] = (df["t_sec"] // bin_sec).astype(int) * bin_sec
    df["t_bin_rel"] = (df["t_rel_sec"] // bin_sec).astype(int) * bin_sec

    # group on meeting-relative bins (so x-axis matches notebook)
    group_key = "t_bin_rel"

    # CD minutes per bin
    cd_bins = (
        df.groupby(group_key)
          .agg(cd_time_sec=("dur_sec", lambda s: float(s[df.loc[s.index, "is_commitment_code"]].sum())))
          .reset_index()
          .rename(columns={group_key: "t_bin"})
          .sort_values("t_bin")
    )
    cd_bins["t_min"] = cd_bins["t_bin"] / 60.0
    cd_bins["cd_min"] = cd_bins["cd_time_sec"] / 60.0

    # Entropy per bin
    ent_rows: list[dict] = []
    for t_bin, g in df.groupby(group_key):
        g2 = g[~g["is_structural_wrap"]].copy() if exclude_structural_from_entropy else g
        toks: list[str] = []
        for txt in g2["text"].tolist():
            toks.extend(_tokenize(txt, stopwords=stopwords))
        ent_rows.append({"t_bin": int(t_bin), "entropy": _shannon_entropy(toks)})

    ent_bins = pd.DataFrame(ent_rows).sort_values("t_bin")

    bins = cd_bins.merge(ent_bins, on="t_bin", how="left").sort_values("t_bin")
    bins = _smooth_cols(bins, cols=["cd_min", "entropy"], smooth_window=smooth_window)

    # naming consistency with plot_entropy_vs_cd()
    bins = bins.rename(columns={"cd_min_smooth": "cd_smooth"})

    return bins, df


# ============================================================
# 3) Slide 7 style: plot entropy vs CD with dual axis
# ============================================================

def plot_entropy_vs_cd(
    bins: pd.DataFrame,
    *,
    session_stem: str = "",
    bin_sec: int = 60,
    plot_structural: bool = False,
    structural_bins: Optional[pd.DataFrame] = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Dual-axis plot:
      left axis: entropy_smooth
      right axis: cd_smooth (and optionally structural_smooth)
    If plot_structural=True, pass structural_bins with columns:
      t_min, structural_smooth   (or t_bin + structural_smooth, but t_min is easiest)

    Returns:
      (fig, ax_entropy, ax_minutes)
    """
    required_cols = {"t_min", "entropy_smooth", "cd_smooth"}
    missing = required_cols - set(bins.columns)
    if missing:
        raise ValueError(f"bins missing required columns: {sorted(missing)}")

    fig, ax_entropy = plt.subplots(figsize=(10, 4))

    # entropy line (left axis)
    ax_entropy.plot(
        bins["t_min"],
        bins["entropy_smooth"],
        color="tab:blue",
        linewidth=2,
        label="Entropy (Shannon; smoothed)",
    )
    ax_entropy.set_xlabel("Time (minutes)")
    ax_entropy.set_ylabel(f"Entropy per {bin_sec}s window (smoothed)")
    ax_entropy.tick_params(axis="y", labelcolor="tab:blue")

    # minutes axis
    ax_minutes = ax_entropy.twinx()
    ax_minutes.plot(
        bins["t_min"],
        bins["cd_smooth"],
        color="tab:orange",
        linewidth=2,
        label=f"Coordination/Decision minutes per {bin_sec}s (smoothed)",
    )
    ax_minutes.set_ylabel(f"Minutes per {bin_sec}s window (smoothed)")
    ax_minutes.tick_params(axis="y", labelcolor="tab:orange")

    # optional structural
    if plot_structural:
        if structural_bins is None:
            raise ValueError("plot_structural=True requires structural_bins.")
        if "t_min" not in structural_bins.columns or "structural_smooth" not in structural_bins.columns:
            raise ValueError("structural_bins must contain columns: t_min, structural_smooth")

        ax_minutes.plot(
            structural_bins["t_min"],
            structural_bins["structural_smooth"],
            color="tab:gray",
            linestyle="--",
            linewidth=2,
            label=f"Structural wrap-up minutes per {bin_sec}s (smoothed)",
        )

    title = f"Entropy vs Coordination/Decision"
    if session_stem:
        title += f" ({session_stem})"
    fig.suptitle(title)

    # combined legend
    lines1, labels1 = ax_entropy.get_legend_handles_labels()
    lines2, labels2 = ax_minutes.get_legend_handles_labels()
    ax_entropy.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.tight_layout()
    return fig, ax_entropy, ax_minutes