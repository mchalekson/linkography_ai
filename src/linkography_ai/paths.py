from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_V2_DIR = REPO_ROOT / "data-v2"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXTERNAL_DATA_V2_DIR = REPO_ROOT.parent / "gemini_data_analysis" / "outputs"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def data_root() -> Path:
    return _env_path("LINKOGRAPHY_AI_DATA_ROOT") or DATA_DIR


def data_v2_root() -> Path:
    return _env_path("LINKOGRAPHY_AI_DATA_V2_ROOT") or (EXTERNAL_DATA_V2_DIR if EXTERNAL_DATA_V2_DIR.exists() else DATA_V2_DIR)


def outputs_root() -> Path:
    return _env_path("LINKOGRAPHY_AI_OUTPUTS_ROOT") or OUTPUTS_DIR


def display_path(path: Path, *, base: Path | None = None) -> str:
    for root in [base, REPO_ROOT]:
        if root is None:
            continue
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)
