from __future__ import annotations
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

def list_conferences() -> List[str]:
    """
    Conferences are directories like 2021MZT, 2021NES, etc.
    Filters out files like all_data_df.xlsx.
    """
    out: List[str] = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "session_data").exists():
            out.append(p.name)
    return out

