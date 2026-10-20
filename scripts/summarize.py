"""Print the per-run results table (Markdown) used in the README.

    python scripts/summarize.py --runs-dir runs

Every number comes from curve.csv, final_eval.csv and meta.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lunarlander_rl.analysis import first_step_reaching, load_curve, load_returns, seed_run_dirs
from lunarlander_rl.evaluation import SUCCESS_THRESHOLD, success_rate

LABELS = {"random": "Random", "dqn": "DQN", "ppo": "PPO"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    print("| Run | Final return, mean ± std (100 ep.) | Success rate | "
          "Best curve point | First step with curve ≥ 200 | Training time |")
    print("|---|---|---|---|---|---|")
    for algo in ("random", "dqn", "ppo"):
        for run_dir in seed_run_dirs(args.runs_dir, algo):
            returns = load_returns(run_dir / "final_eval.csv")
            cells = [f"{LABELS[algo]} {run_dir.name.replace('_', ' ')}",
                     f"{returns.mean():.1f} ± {returns.std(ddof=1):.1f}",
                     f"{success_rate(returns):.0%}"]
            if algo == "random":
                cells += ["–", "–", "–"]
            else:
                curve = load_curve(run_dir / "curve.csv")
                best = int(np.argmax(curve.mean_returns))
                first = first_step_reaching(curve, SUCCESS_THRESHOLD)
                meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
                cells += [f"{curve.mean_returns[best]:.0f} at {curve.timesteps[best] / 1e3:.0f}k",
                          f"{first / 1e3:.0f}k" if first is not None else "never",
                          f"{meta['wall_seconds_train'] / 60:.0f} min"]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
