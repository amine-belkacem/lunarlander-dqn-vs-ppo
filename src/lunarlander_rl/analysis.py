"""Read run outputs and compute the per-seed quantities used in figures.

Everything here works from the raw per-episode CSVs written during training
and final evaluation, so any number in a figure can be traced back to them.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Curve:
    """Learning curve of one run: mean evaluation return at each step."""
    timesteps: np.ndarray
    mean_returns: np.ndarray


def load_curve(curve_csv: Path) -> Curve:
    """Average the evaluation episodes of each evaluation point."""
    by_step: dict[int, list[float]] = {}
    with Path(curve_csv).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_step.setdefault(int(row["timesteps"]), []).append(float(row["return"]))
    if not by_step:
        raise ValueError(f"{curve_csv} has no evaluation rows")
    steps = np.array(sorted(by_step))
    means = np.array([np.mean(by_step[step]) for step in steps])
    return Curve(timesteps=steps, mean_returns=means)


def load_returns(eval_csv: Path) -> np.ndarray:
    with Path(eval_csv).open(encoding="utf-8") as handle:
        returns = [float(row["return"]) for row in csv.DictReader(handle)]
    if not returns:
        raise ValueError(f"{eval_csv} has no episodes")
    return np.array(returns)


def first_step_reaching(curve: Curve, threshold: float) -> int | None:
    """First evaluation step whose mean return is >= threshold, None if never."""
    reached = np.nonzero(curve.mean_returns >= threshold)[0]
    return int(curve.timesteps[reached[0]]) if reached.size else None


def snap_to_grid(timesteps: np.ndarray, every: int) -> np.ndarray:
    """Round evaluation steps to the nominal grid.

    PPO collects 16 steps at a time, so its evaluation at "25,000" really
    happens at 25,008. Rounding to the grid lets seeds be averaged point by
    point; the shift is at most n_envs - 1 steps, invisible at this scale.
    """
    return (np.rint(np.asarray(timesteps) / every) * every).astype(int)


def mean_over_seeds(curves: list[Curve], every: int) -> Curve:
    """Point-by-point mean over seeds, on grid points that all seeds have."""
    if not curves:
        raise ValueError("need at least one curve")
    snapped = [dict(zip(snap_to_grid(c.timesteps, every), c.mean_returns)) for c in curves]
    common = sorted(set.intersection(*(set(s) for s in snapped)))
    means = np.array([np.mean([s[step] for s in snapped]) for step in common])
    return Curve(timesteps=np.array(common), mean_returns=means)


def seed_run_dirs(runs_dir: Path, algo: str) -> list[Path]:
    """runs/<algo>/seed_<k> directories sorted by seed number."""
    dirs = [d for d in (Path(runs_dir) / algo).glob("seed_*") if d.is_dir()]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))
