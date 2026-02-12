from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Utterance:
    text: str
    cdp: List[str]


def iter_session_files(conference: str) -> Iterator[Path]:
    session_dir = DATA_DIR / conference / "session_data"
    if not session_dir.exists():
        return iter(())
    yield from sorted(session_dir.glob("*.json"))


def load_session_outcomes(conference: str) -> Dict[str, Any]:
    path = DATA_DIR / conference / f"{conference}_session_outcomes.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text())

    # Common case: dict keyed by session_id
    if isinstance(obj, dict):
        return obj

    # If list, try to map by session id-like fields
    if isinstance(obj, list):
        out: Dict[str, Any] = {}
        for item in obj:
            if isinstance(item, dict):
                sid = item.get("session_id") or item.get("session") or item.get("id")
                if sid:
                    out[str(sid)] = item
        return out

    return {}


def _extract_cdp_from_utterance_dict(u: Dict[str, Any]) -> List[str]:
    # Direct list fields
    for k in ["cdp", "CDP", "coordination_decision_practices"]:
        v = u.get(k)
        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]

    # Nested dict patterns
    ad = u.get("annotation_dict") or u.get("annotations") or {}
    if isinstance(ad, dict):
        v = (
            ad.get("Coordination and Decision Practices")
            or ad.get("CDP")
            or ad.get("cdp")
        )
        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]

    return []


def load_session_utterances(session_path: Path) -> List[Utterance]:
    obj = json.loads(session_path.read_text())

    if isinstance(obj, dict) and isinstance(obj.get("utterances"), list):
        utter_list = obj["utterances"]
    elif isinstance(obj, list):
        utter_list = obj
    else:
        utter_list = []

    out: List[Utterance] = []
    for u in utter_list:
        if not isinstance(u, dict):
            continue
        text = str(u.get("transcript") or u.get("text") or u.get("utterance") or "")
        cdp = _extract_cdp_from_utterance_dict(u)
        out.append(Utterance(text=text, cdp=cdp))

    return out

