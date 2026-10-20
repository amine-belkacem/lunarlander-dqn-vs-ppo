"""Seeded evaluation of any policy on LunarLander.

I do not use SB3's evaluate_policy here, for two reasons: it does not let me
choose the seed of each episode, and it only works with SB3 models. With
run_episodes, episode i always starts from env.reset(seed=base_seed + i), so
every policy (DQN, PPO, random, later an ONNX model on the Pi) faces exactly
the same initial states.
"""

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from lunarlander_rl.envs import N_ACTIONS, make_env

# LunarLander is considered solved at an average return of 200.
SUCCESS_THRESHOLD = 200.0

Policy = Callable[[np.ndarray], int]


@dataclass(frozen=True)
class EpisodeResults:
    seeds: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def success_rate(self) -> float:
        return success_rate(self.returns)


def success_rate(returns: np.ndarray, threshold: float = SUCCESS_THRESHOLD) -> float:
    """Fraction of episodes whose return reaches the threshold (inclusive)."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        raise ValueError("success_rate needs at least one episode")
    return float(np.mean(returns >= threshold))


def run_episodes(
    policy: Policy,
    n_episodes: int,
    base_seed: int,
    env_factory: Callable[[], gym.Env] = make_env,
) -> EpisodeResults:
    """Play n_episodes, episode i starting from reset(seed=base_seed + i)."""
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    env = env_factory()
    seeds = np.arange(base_seed, base_seed + n_episodes)
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    try:
        for i, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            done = False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                returns[i] += float(reward)
                lengths[i] += 1
                done = terminated or truncated
    finally:
        env.close()
    return EpisodeResults(seeds=seeds, returns=returns, lengths=lengths)


def sb3_policy(model) -> Policy:
    """Wrap an SB3 model as a deterministic obs -> action function."""

    def act(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


class RandomPolicy:
    """Uniform random actions from its own seeded generator."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(N_ACTIONS))
