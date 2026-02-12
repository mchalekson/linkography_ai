from __future__ import annotations
from typing import List


def segment_thirds(n: int) -> List[str]:
    """
    Assign each utterance index into beginning/middle/end by thirds.
    """
    if n <= 0:
        return []
    a = n // 3
    b = (2 * n) // 3
    labels: List[str] = []
    for i in range(n):
        if i < a:
            labels.append("beginning")
        elif i < b:
            labels.append("middle")
        else:
            labels.append("end")
    return labels

