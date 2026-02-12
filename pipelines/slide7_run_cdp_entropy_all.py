from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from linkography_ai.discovery import list_conferences
from linkography_ai.io_sessions import iter_session_files, load_session_outcomes, load_session_utterances
from linkography_ai.segmentation import segment_thirds
from linkography_ai.entropy import shannon_entropy_from_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
TABLES = OUT / "tables"
LOGS = OUT / "logs"

def compute_for_conference(conference: str, normalize: bool, max_sessions: int) -> pd.DataFrame:
    outcomes = load_session_outcomes(conference)

    rows = []
    session_paths = list(iter_session_files(conference))
    if max_sessions and max_sessions > 0:
        session_paths = session_paths[:max_sessions]

    for sp in session_paths:
        utts = load_session_utterances(sp)
        n = len(utts)
        if n == 0:
            continue

        seg_labels = segment_thirds(n)

        seg_counts = { "beginning": Counter(), "middle": Counter(), "end": Counter() }
        for u, seg in zip(utts, seg_labels):
            for code in u.cdp:
                seg_counts[seg][code] += 1

        ent = {}
        for seg in ["beginning", "middle", "end"]:
            counts = list(seg_counts[seg].values())
            ent[f"entropy_{seg}"] = shannon_entropy_from_counts(counts, normalize=normalize) if counts else float("nan")
            ent[f"n_cdp_{seg}"] = int(sum(counts)) if counts else 0
            ent[f"n_unique_cdp_{seg}"] = int(len(seg_counts[seg]))

        sid = sp.stem
        out_payload = outcomes.get(sid, {})
        outcome = None
        if isinstance(out_payload, dict):
            outcome = out_payload.get("outcome") or out_payload.get("label") or out_payload.get("success")

        rows.append({
            "conference": conference,
            "session_id": sid,
            "n_utterances": n,
            "outcome": outcome,
            **ent,
        })

    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference", default="ALL", help="e.g., 2021MZT or ALL")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max_sessions", type=int, default=0, help="0 = all sessions per conference")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.conference.upper() == "ALL":
        conferences = list_conferences()
    else:
        conferences = [args.conference]

    dfs = []
    for conf in conferences:
        df_conf = compute_for_conference(conf, normalize=args.normalize, max_sessions=args.max_sessions)
        dfs.append(df_conf)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = TABLES / f"cdp_entropy_by_session_{args.conference}_{ts}.csv"
    df.to_csv(out_path, index=False)

    log_path = LOGS / f"run_cdp_entropy_{args.conference}_{ts}.txt"
    with open(log_path, "w") as f:
        f.write(f"conference_arg={args.conference}\n")
        f.write(f"conferences_run={conferences}\n")
        f.write(f"sessions_total={len(df)}\n")
        f.write(f"output_table={out_path}\n")

    print(f"Wrote: {out_path}")
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()

