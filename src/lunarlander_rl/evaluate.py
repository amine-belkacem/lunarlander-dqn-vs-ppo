"""Final evaluation of a trained run on held-out episode seeds.

    python -m lunarlander_rl.evaluate --run-dir runs/ppo/seed_0

Training envs are seeded with 0..24 (seed + env index) and then keep drawing
start states from their own random streams. The learning curve resets with
seeds 10000..10009. The final evaluation uses seeds from 1,000,000 upwards,
a separate set that was never used to look at results during training.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from stable_baselines3 import DQN, PPO

from lunarlander_rl.evaluation import RandomPolicy, run_episodes, sb3_policy

FINAL_EVAL_BASE_SEED = 1_000_000
FINAL_EVAL_EPISODES = 100
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}


def write_results(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode_seed", "return", "length"))
        for seed, ret, length in zip(results.seeds, results.returns, results.lengths):
            writer.writerow((int(seed), f"{ret:.4f}", int(length)))


def algo_from_run_dir(run_dir: Path) -> str:
    """Read the algorithm from the run's meta.json.

    The directory name is the config name (dqn_big_buffer, ...), not
    necessarily the algorithm, so it cannot be used for this.
    """
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"{meta_path} not found, is {run_dir} a training run?")
    algo = json.loads(meta_path.read_text(encoding="utf-8"))["config"]["algo"]
    if algo not in ALGO_CLASSES:
        raise ValueError(f"{meta_path}: unknown algorithm {algo!r}")
    return algo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path, help="evaluate final_model.zip of this run")
    target.add_argument("--random", type=int, metavar="SEED",
                        help="evaluate a uniform random policy (baseline)")
    parser.add_argument("--episodes", type=int, default=FINAL_EVAL_EPISODES)
    parser.add_argument("--base-seed", type=int, default=FINAL_EVAL_BASE_SEED)
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: <run-dir>/final_eval.csv)")
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        model_path = args.run_dir / "final_model.zip"
        if not model_path.exists():
            parser.error(f"{model_path} not found, is the run finished?")
        model = ALGO_CLASSES[algo_from_run_dir(args.run_dir)].load(model_path, device="cpu")
        policy = sb3_policy(model)
        out = args.out or args.run_dir / "final_eval.csv"
        label = str(args.run_dir)
    else:
        policy = RandomPolicy(seed=args.random)
        out = args.out or Path("runs") / "random" / f"seed_{args.random}" / "final_eval.csv"
        label = f"random policy (seed {args.random})"

    results = run_episodes(policy, n_episodes=args.episodes, base_seed=args.base_seed)
    write_results(out, results)
    print(f"{label}: mean return {results.mean_return:.1f}, "
          f"success rate {results.success_rate:.0%} over {args.episodes} episodes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
