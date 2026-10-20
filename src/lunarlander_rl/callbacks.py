"""Training callback: learning curve on a fixed step grid, plus snapshots."""

import csv
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from lunarlander_rl.config import EvaluationSettings
from lunarlander_rl.evaluation import run_episodes, sb3_policy

CURVE_COLUMNS = ("timesteps", "episode_seed", "return", "length")


class LearningCurveCallback(BaseCallback):
    """Evaluate the current policy every `every_timesteps` environment steps.

    SB3 counts callback calls per vectorized step, so with 16 PPO envs a
    call-based frequency would mean 16x more steps than with DQN. I trigger on
    model.num_timesteps instead, which is the same unit for both algorithms.
    The same episode seeds are reused at every evaluation point, so changes in
    the curve come from the policy, not from easier or harder start states.

    Every episode is written to CSV (long format), not just the mean, so the
    statistics in J2 can be recomputed from raw data.
    """

    def __init__(self, settings: EvaluationSettings, curve_path: Path,
                 checkpoint_dir: Path, checkpoints: tuple[int, ...] = ()):
        super().__init__(verbose=0)
        self.settings = settings
        self.curve_path = Path(curve_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._pending_checkpoints = sorted(checkpoints)
        self._next_eval = 0
        self.eval_seconds = 0.0

    def _on_training_start(self) -> None:
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        with self.curve_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CURVE_COLUMNS)
        # Point 0: the untrained policy, so every curve starts from the same place.
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
        while self._pending_checkpoints and self.num_timesteps >= self._pending_checkpoints[0]:
            target = self._pending_checkpoints.pop(0)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_dir / f"step_{target}")
        return True

    def _evaluate(self) -> None:
        start = time.perf_counter()
        results = run_episodes(
            sb3_policy(self.model),
            n_episodes=self.settings.episodes,
            base_seed=self.settings.base_seed,
        )
        self.eval_seconds += time.perf_counter() - start

        with self.curve_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for seed, ret, length in zip(results.seeds, results.returns, results.lengths):
                writer.writerow((self.num_timesteps, int(seed), f"{ret:.4f}", int(length)))

        # Next grid point strictly above the current step count. PPO advances
        # 16 steps at a time, so it can land a few steps past a grid point.
        every = self.settings.every_timesteps
        self._next_eval = (self.num_timesteps // every + 1) * every
