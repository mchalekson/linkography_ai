from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


_CHUNK_PAT = re.compile(r"^(?P<base>.+)_chunk(?P<chunk>\d+)$")
_TOKEN_PAT = re.compile(r"[a-z0-9']+")


@dataclass
class Move:
    speaker: str
    text: str
    start_sec: float
    end_sec: float
    chunk_index: int
    utterance_index: int


def parse_time_to_seconds(ts: Any) -> float:
    if not isinstance(ts, str) or ":" not in ts:
        return math.nan
    parts = ts.split(":")
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return math.nan
    if len(nums) == 2:
        mm, ss = nums
        return mm * 60 + ss
    if len(nums) == 3:
        hh, mm, ss = nums
        return hh * 3600 + mm * 60 + ss
    return math.nan


def parse_timestamp_range(ts: Any) -> Tuple[float, float]:
    if not isinstance(ts, str) or "-" not in ts:
        return math.nan, math.nan
    start_s, end_s = ts.split("-", 1)
    start = parse_time_to_seconds(start_s.strip())
    end = parse_time_to_seconds(end_s.strip())
    return start, end


def _base_from_chunk_filename(path: Path) -> Tuple[str, int]:
    m = _CHUNK_PAT.match(path.stem)
    if not m:
        return path.stem, 0
    return m.group("base"), int(m.group("chunk"))


def _build_move_text(utt: Dict[str, Any]) -> str:
    direct = utt.get("transcript") or utt.get("text") or utt.get("utterance")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    codes = utt.get("codes")
    if not isinstance(codes, list):
        return ""

    snippets: List[str] = []
    seen: set[str] = set()
    for c in codes:
        if not isinstance(c, dict):
            continue
        evidence = c.get("evidence")
        if not isinstance(evidence, str):
            continue
        text = evidence.strip()
        if not text or text.lower() == "none":
            continue
        if text in seen:
            continue
        seen.add(text)
        snippets.append(text)

    return " ".join(snippets)


def load_chunk_moves(chunk_path: Path, chunk_index: int) -> List[Move]:
    try:
        obj = json.loads(chunk_path.read_text())
    except Exception:
        return []
    utts = obj.get("utterance_annotations")
    if not isinstance(utts, list):
        return []

    moves: List[Move] = []
    for i, utt in enumerate(utts):
        if not isinstance(utt, dict):
            continue
        speaker = str(utt.get("speaker") or "")
        start_sec, end_sec = parse_timestamp_range(utt.get("timestamp"))
        if math.isnan(start_sec):
            continue
        if math.isnan(end_sec):
            dur = utt.get("speaking_duration_seconds")
            if isinstance(dur, (int, float)):
                end_sec = float(start_sec + max(0.0, float(dur)))
            else:
                end_sec = start_sec
        if end_sec < start_sec:
            end_sec = start_sec

        text = _build_move_text(utt)

        moves.append(
            Move(
                speaker=speaker,
                text=text,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                chunk_index=chunk_index,
                utterance_index=i,
            )
        )

    moves.sort(key=lambda m: (m.start_sec, m.utterance_index))
    return moves


def group_v2_chunk_files(outputs_v2_root: Path) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for chunk_fp in outputs_v2_root.rglob("*_chunk*.json"):
        rel = chunk_fp.relative_to(outputs_v2_root)
        if len(rel.parts) < 2:
            continue
        conference = rel.parts[0]
        session_folder = rel.parts[1]
        base, chunk_i = _base_from_chunk_filename(chunk_fp)

        meeting_id = f"{session_folder}__{base}"
        key = f"{conference}::{meeting_id}"
        if key not in grouped:
            grouped[key] = {
                "conference": conference,
                "session_id": session_folder,
                "meeting_id": meeting_id,
                "chunk_files": [],
            }
        grouped[key]["chunk_files"].append((chunk_i, chunk_fp))

    for group in grouped.values():
        group["chunk_files"].sort(key=lambda x: (x[0], str(x[1])))
    return grouped


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PAT.findall(text.lower())


def tfidf_cosine_similarity(texts: List[str]) -> np.ndarray:
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.eye(1, dtype=float)

    tokenized = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for toks in tokenized:
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    if not vocab:
        return np.eye(n, dtype=float)

    vsize = len(vocab)
    tf = np.zeros((n, vsize), dtype=float)
    df = np.zeros(vsize, dtype=float)

    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        counts: Dict[int, int] = {}
        for tok in toks:
            j = vocab[tok]
            counts[j] = counts.get(j, 0) + 1
        length = float(len(toks))
        for j, c in counts.items():
            tf[i, j] = c / length
        for j in counts:
            df[j] += 1.0

    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    x = tf * idf

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    sim = x @ x.T
    np.fill_diagonal(sim, 1.0)
    return sim


def fuzzy_weight_matrix(sim: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    if sim.size == 0:
        return sim.copy()
    t = max(0.0, min(0.999999, float(threshold)))
    w = (sim - t) / (1.0 - t)
    w = np.clip(w, 0.0, 1.0)
    np.fill_diagonal(w, 0.0)
    return w


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def compute_fuzzy_metrics(moves: List[Move], threshold: float = 0.35) -> Dict[str, float]:
    n = len(moves)
    out: Dict[str, float] = {
        "n_moves": float(n),
        "threshold": float(threshold),
    }
    if n <= 1:
        out.update(
            {
                "n_possible_links": 0.0,
                "n_nonzero_links": 0.0,
                "total_link_weight": 0.0,
                "weighted_ldi": 0.0,
                "mean_nonzero_weight": 0.0,
                "forelink_weight_mean": 0.0,
                "backlink_weight_mean": 0.0,
                "cross_speaker_weight_ratio": 0.0,
                "late_minus_early_backlink": 0.0,
                "forelink_entropy": 0.0,
                "backlink_entropy": 0.0,
                "horizon_entropy": 0.0,
                "overall_link_entropy": 0.0,
            }
        )
        return out

    texts = [m.text for m in moves]
    sim = tfidf_cosine_similarity(texts)
    w = fuzzy_weight_matrix(sim, threshold=threshold)

    upper = np.triu(w, k=1)
    n_possible = n * (n - 1) / 2
    nz = float((upper > 0).sum())
    total_w = float(upper.sum())
    mean_nonzero = float(total_w / nz) if nz > 0 else 0.0

    fore = upper.sum(axis=1)
    back = upper.sum(axis=0)

    speakers = [m.speaker for m in moves]
    cross_w = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if speakers[i] != speakers[j]:
                cross_w += float(w[i, j])
    cross_ratio = cross_w / total_w if total_w > 0 else 0.0

    starts = np.array([m.start_sec for m in moves], dtype=float)
    t_min = float(np.min(starts))
    t_max = float(np.max(starts))
    t_len = max(1.0, t_max - t_min)
    first_cut = t_min + t_len / 3.0
    second_cut = t_min + 2.0 * t_len / 3.0
    early_idx = [i for i, t in enumerate(starts) if t <= first_cut]
    late_idx = [i for i, t in enumerate(starts) if t >= second_cut]
    early_back = float(np.mean(back[early_idx])) if early_idx else 0.0
    late_back = float(np.mean(back[late_idx])) if late_idx else 0.0

    fore_ent = 0.0
    for i in range(n - 1):
        possible = n - i - 1
        p = float(fore[i] / possible) if possible > 0 else 0.0
        fore_ent += _binary_entropy(max(0.0, min(1.0, p)))

    back_ent = 0.0
    for j in range(1, n):
        possible = j
        p = float(back[j] / possible) if possible > 0 else 0.0
        back_ent += _binary_entropy(max(0.0, min(1.0, p)))

    horizon_ent = 0.0
    for d in range(1, n):
        vals = [w[i, i + d] for i in range(n - d)]
        if not vals:
            continue
        p = float(np.mean(vals))
        horizon_ent += _binary_entropy(max(0.0, min(1.0, p)))

    out.update(
        {
            "n_possible_links": float(n_possible),
            "n_nonzero_links": float(nz),
            "total_link_weight": float(total_w),
            "weighted_ldi": float(total_w / n),
            "mean_nonzero_weight": mean_nonzero,
            "forelink_weight_mean": float(np.mean(fore)),
            "backlink_weight_mean": float(np.mean(back)),
            "cross_speaker_weight_ratio": float(cross_ratio),
            "late_minus_early_backlink": float(late_back - early_back),
            "forelink_entropy": float(fore_ent),
            "backlink_entropy": float(back_ent),
            "horizon_entropy": float(horizon_ent),
            "overall_link_entropy": float(fore_ent + back_ent + horizon_ent),
        }
    )
    return out
