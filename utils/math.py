"""Lightweight math helpers for examples and notebooks."""

import math
from typing import Iterable, List


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def softmax(logits: Iterable[float]) -> List[float]:
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]
