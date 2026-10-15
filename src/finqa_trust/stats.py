"""Uncertainty for accuracy numbers: bootstrap intervals and sample sizes.

Two kinds of noise matter. Model randomness shows up when the same questions are run again.
Question sampling, which questions happened to be drawn, does not; the bootstrap measures it
by resampling the questions we have.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks (q in [0, 1])."""
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(correct_flags: list[bool | int], n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Accuracy and its percentile-bootstrap interval: (accuracy, low, high).

    Each round draws len(flags) questions with replacement and keeps the accuracy of the draw.
    """
    n = len(correct_flags)
    if n == 0:
        raise ValueError("no answers")
    flags = [int(f) for f in correct_flags]
    rng = random.Random(seed)
    stats = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    tail = (1 - level) / 2
    return sum(flags) / n, _percentile(stats, tail), _percentile(stats, 1 - tail)


def paired_bootstrap_diff(a_flags: list[bool | int], b_flags: list[bool | int], n_boot: int = 2000,
                          seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Accuracy of B minus accuracy of A on the same questions, with its interval.

    The same resampled questions are used for both in every round, so question difficulty
    cancels out of the difference. If the interval excludes 0, the difference is real at this level.
    """
    if len(a_flags) != len(b_flags):
        raise ValueError("paired comparison needs the same questions in the same order")
    diffs = [int(b) - int(a) for a, b in zip(a_flags, b_flags)]
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    tail = (1 - level) / 2
    return sum(diffs) / n, _percentile(stats, tail), _percentile(stats, 1 - tail)


def n_for_difference(delta: float, accuracy: float | None = None, discordant: float | None = None,
                     alpha: float = 0.05, power: float = 0.8) -> int:
    """Questions needed to detect a true accuracy difference of `delta` (e.g. 0.03).

    Unpaired (two separate question sets), give `accuracy`: n per approach.
    Paired (same questions), give `discordant`: the share of questions where the two approaches
    disagree (one right, one wrong). Fewer disagreements -> far fewer questions needed.
    """
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    if discordant is not None:
        return math.ceil(z ** 2 * discordant / delta ** 2)
    if accuracy is None:
        raise ValueError("give accuracy (unpaired) or discordant (paired)")
    return math.ceil(2 * z ** 2 * accuracy * (1 - accuracy) / delta ** 2)


def ci_half_width(accuracy: float, n: int, level: float = 0.95) -> float:
    """Normal-approximation half width of the interval for one accuracy on n questions."""
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    return z * math.sqrt(accuracy * (1 - accuracy) / n)
