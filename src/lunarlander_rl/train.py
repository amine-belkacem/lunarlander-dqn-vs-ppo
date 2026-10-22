"""Train one (algorithm, seed) run from a YAML config.

    python -m lunarlander_rl.train --config configs/ppo.yaml --seed 0

Outputs go to <out-dir>/<config name>/seed_<seed>/, where the config name is
the YAML file name (dqn, ppo, dqn_big_buffer...), so a variant never
overwrites the baseline runs of the same algorithm:
    curve.csv        every evaluation episode during training
    final_model.zip  policy at the end of training
    checkpoints/     snapshots listed in the config (for the video)
    meta.json        config, versions, git commit, wall-clock times
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from lunarlander_rl.callbacks import LearningCurveCallback
from lunarlander_rl.config import RunConfig, load_config
from lunarlander_rl.envs import ENV_ID

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}


def run_dir_for(out_dir: Path, name: str, seed: int) -> Path:
    return Path(out_dir) / name / f"seed_{seed}"


def _git_commit() -> str:
    """Commit of the code that produced the run, with '-dirty' if modified."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(config: RunConfig, seed: int, out_dir: Path, threads: int = 1,
          name: str | None = None) -> Path:
    """Train one run. `name` defaults to the algorithm (runs/<algo>/seed_k)."""
    name = name or config.algo
    torch.set_num_threads(threads)
    run_dir = run_dir_for(out_dir, name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "env_id": ENV_ID,
        "config_name": name,
        "seed": seed,
        "config": config.to_dict(),
        "torch_threads": threads,
        "git_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "status": "running",
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Env i of the vector gets seed + i; the model seed fixes torch/numpy/random.
    env = make_vec_env(ENV_ID, n_envs=config.n_envs, seed=seed)
    model = ALGO_CLASSES[config.algo](
        "MlpPolicy",
        env,
        policy_kwargs=config.policy_kwargs or None,
        seed=seed,
        device="cpu",
        verbose=0,
        **config.hyperparams,
    )
    callback = LearningCurveCallback(
        settings=config.evaluation,
        curve_path=run_dir / "curve.csv",
        checkpoint_dir=run_dir / "checkpoints",
        checkpoints=config.checkpoints,
    )

    start = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    wall_seconds = time.perf_counter() - start
    model.save(run_dir / "final_model")
    env.close()

    meta.update({
        "status": "done",
        "actual_timesteps": int(model.num_timesteps),
        "wall_seconds_total": round(wall_seconds, 1),
        "wall_seconds_eval": round(callback.eval_seconds, 1),
        "wall_seconds_train": round(wall_seconds - callback.eval_seconds, 1),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads for this run (default 1)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="override the config budget, for timing or smoke tests only")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.total_timesteps is not None:
        config = replace(config, total_timesteps=args.total_timesteps)

    run_dir = train(config, seed=args.seed, out_dir=args.out_dir, threads=args.threads,
                    name=args.config.stem)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"{config.algo} seed {args.seed}: {meta['actual_timesteps']} steps in "
          f"{meta['wall_seconds_total']} s (eval {meta['wall_seconds_eval']} s) -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
