"""Aggregate statistics over training seeds.

Following Agarwal et al., "Deep RL at the Edge of the Statistical Precipice"
(NeurIPS 2021), the unit of resampling is the training run, not the episode:
100 episodes of one run say how good that run is, but only more runs say how
reliable the algorithm is. Every function here therefore takes one value per
run.

I implement IQM and the bootstrap myself instead of depending on rliable. The
IQM definition is the same as rliable's (scipy.stats.trim_mean with 25% cut
on each side); with a single task, rliable's stratified bootstrap reduces to
the plain percentile bootstrap over runs used here.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_N_BOOTSTRAP = 50_000
DEFAULT_CONFIDENCE = 0.95


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: drop the lowest and highest 25%, average the rest.

    As in scipy.stats.trim_mean, int(0.25 * n) values are cut from each end.
    Less sensitive than the mean to one catastrophic run, more efficient
    than the median.
    """
    values = np.sort(np.asarray(values, dtype=float))
    n = values.size
    if n == 0:
        raise ValueError("iqm needs at least one value")
    cut = int(0.25 * n)
    return float(np.mean(values[cut:n - cut]))


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.1f} [{self.low:.1f}, {self.high:.1f}]"


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = iqm,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap confidence interval of statistic(values).

    Draws n_bootstrap samples of the runs with replacement, computes the
    statistic on each, and returns the central `confidence` interval of
    those values. Seeded, so the published interval is reproducible.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    resamples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    estimates = np.array([statistic(sample) for sample in resamples])
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail])
    return Estimate(point=statistic(values), low=float(low), high=float(high))


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) for a random run of X against a random run of Y, ties count 1/2.

    This is the Mann-Whitney U statistic divided by len(x) * len(y). It
    answers "if I train once with each algorithm, how likely is X to win?",
    which is closer to what a practitioner wants than a difference of means.
    """
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    if x.size == 0 or y.size == 0:
        raise ValueError("probability_of_improvement needs runs on both sides")
    return float(np.mean((x > y) + 0.5 * (x == y)))
