#!/usr/bin/env python3
"""Cross-check old CDP session JSONs against new v2 chunk JSONs.

Default scope replicates the verification run:
- Conferences: 2021CMC, 2020NES
- Old JSON root: linkography_ai/data/<conference>/session_data/*.json
- New JSON root: linkography_ai/data-v2/<conference>/output_<session_id>/...chunk*.json

Outputs are written into this folder:
- verification_session_metrics.csv
- verification_conference_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_time(t: str | None) -> int | None:
    if not t:
        return None
    parts = t.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return None


def safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def old_label(score2_share: float | None) -> str:
    # Heuristic proxy to compare old CDP bins to new chunk trajectory labels.
    if score2_share is None:
        return "procedural"
    if score2_share >= 0.55:
        return "convergent"
    if score2_share <= 0.45:
        return "divergent"
    return "ambiguous"


def find_gemini_chunks(gem_session_dir: Path) -> list[tuple[int | None, str | None, int | str | None, Path]]:
    chunk_files = sorted(gem_session_dir.rglob("*chunk*.json"))

    if not chunk_files:
        # fallback for directories where chunk files are nested with other names
        candidates = sorted(gem_session_dir.rglob("*.json"))
        filtered: list[Path] = []
        for p in candidates:
            try:
                obj = json.loads(p.read_text())
                if "chunk_summary" in obj:
                    filtered.append(p)
            except Exception:
                continue
        chunk_files = filtered

    chunks: list[tuple[int | None, str | None, int | str | None, Path]] = []
    for p in chunk_files:
        try:
            obj = json.loads(p.read_text())
            cs = obj.get("chunk_summary", {})
            traj = cs.get("idea_trajectory")
            decision = cs.get("decision_crystallization_level")
            m = re.search(r"chunk(\d+)", p.stem)
            idx = int(m.group(1)) if m else None
            chunks.append((idx, traj, decision, p))
        except Exception:
            continue

    return chunks


def analyze_session(old_session_path: Path, gem_session_dir: Path, conference: str, session_id: str) -> dict[str, Any] | None:
    old_obj = json.loads(old_session_path.read_text())
    utterances = old_obj.get("all_data", [])

    chunks = find_gemini_chunks(gem_session_dir)
    if not chunks:
        return None

    # Order chunks deterministically
    if any(c[0] is not None for c in chunks):
        chunks.sort(key=lambda x: (10**9 if x[0] is None else x[0], str(x[3])))
    else:
        chunks.sort(key=lambda x: str(x[3]))

    # Reindex to contiguous 1..N
    chunks = [(i + 1, t, d, p) for i, (_, t, d, p) in enumerate(chunks)]
    n_chunks = len(chunks)

    # Determine old session duration
    max_t = 0
    for u in utterances:
        st = parse_time(u.get("start_time"))
        et = parse_time(u.get("end_time"))
        dur = u.get("speaking_duration")
        if et is not None:
            max_t = max(max_t, et)
        elif st is not None and isinstance(dur, (int, float)):
            max_t = max(max_t, st + int(dur))

    if max_t <= 0:
        return None

    bin_width = max_t / n_chunks
    bins = {i: {"s1": 0, "s2": 0, "tot": 0} for i in range(1, n_chunks + 1)}

    # Aggregate old CDP scores into same number of bins as Gemini chunks
    for u in utterances:
        st = parse_time(u.get("start_time"))
        if st is None:
            continue
        b = min(n_chunks, max(1, int(st // bin_width) + 1))

        ann = u.get("annotations", {})
        cdp = ann.get("Coordination and Decision Practices") if isinstance(ann, dict) else None
        if isinstance(cdp, dict):
            score = cdp.get("score")
            if score in (1, 2):
                bins[b]["tot"] += 1
                if score == 1:
                    bins[b]["s1"] += 1
                else:
                    bins[b]["s2"] += 1

    xs: list[float] = []
    ys: list[float] = []
    considered = 0
    matches = 0
    cdp_bins = 0
    sequence: list[str | None] = []

    for i, (_, traj, decision, _) in enumerate(chunks, start=1):
        sequence.append(traj)
        d = bins[i]
        if d["tot"] > 0:
            cdp_bins += 1
            s2_share = d["s2"] / d["tot"]
            if isinstance(decision, int):
                xs.append(s2_share)
                ys.append(float(decision))
        else:
            s2_share = None

        lbl = old_label(s2_share)
        if lbl != "ambiguous":
            considered += 1
            if lbl == traj:
                matches += 1

    corr = safe_corr(xs, ys)
    switches = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])
    first_conv = next((i + 1 for i, t in enumerate(sequence) if t == "convergent"), -1)

    return {
        "conference": conference,
        "session_id": session_id,
        "n_chunks": n_chunks,
        "cdp_bins_with_data": cdp_bins,
        "cdp_coverage": round(cdp_bins / n_chunks, 3),
        "heuristic_match_rate": round(matches / considered, 3) if considered else math.nan,
        "corr_s2share_vs_decision": round(corr, 3) if corr == corr else math.nan,
        "switches": switches,
        "first_convergent_chunk": first_conv,
        "gemini_sequence": ">".join(str(x) for x in sequence),
    }


def run(old_root: Path, gem_root: Path, conferences: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for conf in conferences:
        old_sessions = {p.stem: p for p in (old_root / conf / "session_data").glob("*.json")}
        gem_dirs = [d for d in (gem_root / conf).glob("output_*") if d.is_dir()]

        matched: list[tuple[str, Path, Path]] = []
        for d in gem_dirs:
            sid = d.name.replace("output_", "")
            if sid in old_sessions:
                matched.append((sid, old_sessions[sid], d))

        conf_rows: list[dict[str, Any]] = []
        for sid, old_fp, gem_dir in sorted(matched):
            r = analyze_session(old_fp, gem_dir, conf, sid)
            if r is not None:
                conf_rows.append(r)
                rows.append(r)

        if conf_rows:
            cov = [r["cdp_coverage"] for r in conf_rows]
            hm = [r["heuristic_match_rate"] for r in conf_rows if r["heuristic_match_rate"] == r["heuristic_match_rate"]]
            cr = [r["corr_s2share_vs_decision"] for r in conf_rows if r["corr_s2share_vs_decision"] == r["corr_s2share_vs_decision"]]
            summary[conf] = {
                "matched_sessions": len(conf_rows),
                "mean_coverage": round(mean(cov), 3),
                "mean_match_rate": round(mean(hm), 3) if hm else None,
                "median_match_rate": round(median(hm), 3) if hm else None,
                "mean_corr": round(mean(cr), 3) if cr else None,
                "median_corr": round(median(cr), 3) if cr else None,
                "low_coverage_lt_0.5": sum(1 for v in cov if v < 0.5),
            }
        else:
            summary[conf] = {"matched_sessions": 0}

    summary["total_matched_sessions"] = len(rows)
    return rows, summary


def main() -> None:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Cross-check old CDP JSON vs new v2 chunk JSON")
    parser.add_argument(
        "--old-root",
        type=Path,
        default=here.parent.parent / "data",
        help="Path to old repo data root (default: ../../data)",
    )
    parser.add_argument(
        "--gem-root",
        type=Path,
        default=here.parent.parent / "data-v2",
        help="Path to v2 chunk annotation root",
    )
    parser.add_argument(
        "--conferences",
        nargs="+",
        default=["2021CMC", "2020NES"],
        help="Conference codes to evaluate",
    )
    args = parser.parse_args()

    rows, summary = run(args.old_root, args.gem_root, args.conferences)

    out_csv = here / "verification_session_metrics.csv"
    out_json = here / "verification_conference_summary.json"

    fieldnames = [
        "conference",
        "session_id",
        "n_chunks",
        "cdp_bins_with_data",
        "cdp_coverage",
        "heuristic_match_rate",
        "corr_s2share_vs_decision",
        "switches",
        "first_convergent_chunk",
        "gemini_sequence",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
