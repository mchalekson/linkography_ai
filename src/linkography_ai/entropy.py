from __future__ import annotations
import math
from typing import List


def shannon_entropy_from_counts(counts: List[int], normalize: bool = False) -> float:
    """
    Shannon entropy H = -sum(p_i log2 p_i)
    If normalize=True, divide by log2(K) where K is number of nonzero categories.
    """
    total = sum(counts)
    if total <= 0:
        return float("nan")

    ps = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p, 2) for p in ps)

    if not normalize:
        return h

    k = len(ps)
    if k <= 1:
        return 0.0
    return h / math.log(k, 2)

