#!/usr/bin/env python
"""Speaker Role and CDP Analysis

Extracts speaker roles (e.g., facilitator, fellow, mentor) from session metadata
and correlates with CDP usage patterns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from linkography_ai.discovery import list_conferences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
ANALYSIS_DIR = OUT_DIR / "analysis"


def extract_speaker_roles(session_json: dict) -> Dict[str, str]:
    """Extract speaker name -> role mapping."""
    roles = {}

    # Try facilitators field
    for fac in session_json.get("facilitators", []):
        roles[fac] = "facilitator"

    # Try all_speakers with role field
    for u in session_json.get("all_data", []):
        if not isinstance(u, dict):
            continue
        speaker = (u.get("speaker") or "").strip()
        if not speaker:
            continue
        role = (u.get("role") or "participant").strip()
        if speaker not in roles:
            roles[speaker] = role

    return roles


def extract_speaker_cdp(session_json: dict) -> Dict[str, dict]:
    """Extract speaker -> CDP score counts."""
    speaker_cdp = {}

    for u in session_json.get("all_data", []):
        if not isinstance(u, dict):
            continue
        speaker = (u.get("speaker") or "").strip()
        if not speaker:
            continue

        ann = u.get("annotations") or u.get("annotation_dict") or {}
        if not isinstance(ann, dict):
            continue

        cdp_data = ann.get("Coordination and Decision Practices")
        if not isinstance(cdp_data, dict):
            continue

        score = cdp_data.get("score")
        if score not in (1, 2):
            continue

        if speaker not in speaker_cdp:
            speaker_cdp[speaker] = {"score1": 0, "score2": 0}
        if score == 1:
            speaker_cdp[speaker]["score1"] += 1
        else:
            speaker_cdp[speaker]["score2"] += 1

    return speaker_cdp


def analyze_session(session_path: Path) -> Optional[dict]:
    try:
        obj = json.loads(session_path.read_text())
    except Exception:
        return None

    session_json = obj if isinstance(obj, dict) and "all_data" in obj else None
    if session_json is None:
        return None

    roles = extract_speaker_roles(session_json)
    speaker_cdp = extract_speaker_cdp(session_json)

    if not speaker_cdp:
        return None

    # Aggregate by role
    role_stats = {}
    for speaker, cdp_counts in speaker_cdp.items():
        role = roles.get(speaker, "participant")
        if role not in role_stats:
            role_stats[role] = {"n_speakers": 0, "score1": 0, "score2": 0}
        role_stats[role]["n_speakers"] += 1
        role_stats[role]["score1"] += cdp_counts["score1"]
        role_stats[role]["score2"] += cdp_counts["score2"]

    # Serialize role stats
    has_facilitator = "facilitator" in role_stats
    facilitator_score1 = role_stats.get("facilitator", {}).get("score1", 0)
    facilitator_score2 = role_stats.get("facilitator", {}).get("score2", 0)
    non_facilitator_score1 = sum(
        v["score1"] for k, v in role_stats.items() if k != "facilitator"
    )
    non_facilitator_score2 = sum(
        v["score2"] for k, v in role_stats.items() if k != "facilitator"
    )

    return {
        "session_id": session_path.stem,
        "has_facilitator": int(has_facilitator),
        "facilitator_cdp_score1": facilitator_score1,
        "facilitator_cdp_score2": facilitator_score2,
        "non_facilitator_cdp_score1": non_facilitator_score1,
        "non_facilitator_cdp_score2": non_facilitator_score2,
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    conferences = list_conferences()
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
    out_path = TABLES_DIR / "speaker_role_cdp.csv"
    df.to_csv(out_path, index=False)

    report_path = ANALYSIS_DIR / "speaker_role_cdp_summary.txt"
    with open(report_path, "w") as f:
        f.write("SPEAKER ROLE AND CDP ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sessions analyzed: {len(df)}\n")
        f.write(f"Sessions with facilitator identified: {df['has_facilitator'].sum()}\n\n")

        fac_subset = df[df["has_facilitator"] == 1]
        if len(fac_subset) > 0:
            f.write("Facilitator CDP (sessions with facilitator):\n")
            f.write(f"  Mean score 1 utterances: {fac_subset['facilitator_cdp_score1'].mean():.2f}\n")
            f.write(f"  Mean score 2 utterances: {fac_subset['facilitator_cdp_score2'].mean():.2f}\n\n")

            f.write("Non-Facilitator CDP (sessions with facilitator):\n")
            f.write(f"  Mean score 1 utterances: {fac_subset['non_facilitator_cdp_score1'].mean():.2f}\n")
            f.write(f"  Mean score 2 utterances: {fac_subset['non_facilitator_cdp_score2'].mean():.2f}\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
