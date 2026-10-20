"""Load and validate the YAML run configurations in configs/."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ALGOS = ("dqn", "ppo")
REQUIRED_KEYS = ("algo", "total_timesteps", "n_envs", "hyperparams", "evaluation")
REQUIRED_EVAL_KEYS = ("every_timesteps", "episodes", "base_seed")


@dataclass(frozen=True)
class EvaluationSettings:
    every_timesteps: int
    episodes: int
    base_seed: int


@dataclass(frozen=True)
class RunConfig:
    algo: str
    total_timesteps: int
    n_envs: int
    hyperparams: dict[str, Any]
    policy_kwargs: dict[str, Any]
    evaluation: EvaluationSettings
    checkpoints: tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "hyperparams": dict(self.hyperparams),
            "policy_kwargs": dict(self.policy_kwargs),
            "evaluation": {
                "every_timesteps": self.evaluation.every_timesteps,
                "episodes": self.evaluation.episodes,
                "base_seed": self.evaluation.base_seed,
            },
            "checkpoints": list(self.checkpoints),
        }


def _positive_int(raw: dict[str, Any], key: str, where: str) -> int:
    value = raw[key]
    # bool is a subclass of int in Python; "true" must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}: '{key}' must be a positive integer, got {value!r}")
    return value


def parse_config(raw: dict[str, Any], source: str = "<dict>") -> RunConfig:
    """Validate a raw config dict and turn it into a RunConfig."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level must be a mapping, got {type(raw).__name__}")

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")

    algo = raw["algo"]
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"{source}: unknown algo {algo!r}, expected one of {SUPPORTED_ALGOS}")

    eval_raw = raw["evaluation"]
    if not isinstance(eval_raw, dict):
        raise ValueError(f"{source}: 'evaluation' must be a mapping")
    missing_eval = [key for key in REQUIRED_EVAL_KEYS if key not in eval_raw]
    if missing_eval:
        raise ValueError(f"{source}: evaluation is missing keys {missing_eval}")

    base_seed = eval_raw["base_seed"]
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError(f"{source}: evaluation.base_seed must be an integer >= 0, got {base_seed!r}")

    evaluation = EvaluationSettings(
        every_timesteps=_positive_int(eval_raw, "every_timesteps", f"{source} evaluation"),
        episodes=_positive_int(eval_raw, "episodes", f"{source} evaluation"),
        base_seed=base_seed,
    )

    hyperparams = raw["hyperparams"]
    if not isinstance(hyperparams, dict):
        raise ValueError(f"{source}: 'hyperparams' must be a mapping")

    policy_kwargs = raw.get("policy_kwargs") or {}
    if not isinstance(policy_kwargs, dict):
        raise ValueError(f"{source}: 'policy_kwargs' must be a mapping")

    checkpoints = raw.get("checkpoints") or []
    if any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in checkpoints):
        raise ValueError(f"{source}: checkpoints must be positive integers, got {checkpoints!r}")

    return RunConfig(
        algo=algo,
        total_timesteps=_positive_int(raw, "total_timesteps", source),
        n_envs=_positive_int(raw, "n_envs", source),
        hyperparams=dict(hyperparams),
        policy_kwargs=dict(policy_kwargs),
        evaluation=evaluation,
        checkpoints=tuple(sorted(checkpoints)),
    )


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw, source=str(path))
