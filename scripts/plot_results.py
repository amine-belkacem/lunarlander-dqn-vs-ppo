"""Draw the learning-curve and final-return figures from the run CSVs.

    python scripts/plot_results.py --runs-dir runs --out-dir figures

No smoothing: each point of a learning curve is the plain mean of the
evaluation episodes recorded at that step.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lunarlander_rl.analysis import load_curve, load_returns, mean_over_seeds, seed_run_dirs
from lunarlander_rl.config import load_config
from lunarlander_rl.evaluation import SUCCESS_THRESHOLD

COLORS = {"dqn": "#1baf7a", "ppo": "#6250d6", "random": "#8a8986"}
LABELS = {"dqn": "DQN", "ppo": "PPO", "random": "Random"}
# Below this, final-return points are not drawn but counted on the figure,
# otherwise a single -5000 episode would squash every other point.
FINAL_RETURN_FLOOR = -1600.0
JITTER_SEED = 0


def plot_learning_curves(runs_dir: Path, out_path: Path, every: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for algo in ("dqn", "ppo"):
        curves = [load_curve(d / "curve.csv") for d in seed_run_dirs(runs_dir, algo)]
        for curve in curves:
            ax.plot(curve.timesteps, curve.mean_returns, color=COLORS[algo],
                    linewidth=0.9, alpha=0.35)
        mean = mean_over_seeds(curves, every)
        ax.plot(mean.timesteps, mean.mean_returns, color=COLORS[algo], linewidth=2.4,
                label=f"{LABELS[algo]}: mean of {len(curves)} seeds (thin lines: each seed)")

    random_returns = np.concatenate(
        [load_returns(d / "final_eval.csv") for d in seed_run_dirs(runs_dir, "random")])
    ax.axhline(random_returns.mean(), color=COLORS["random"], linestyle=":", linewidth=1.5)
    ax.text(ax.get_xlim()[1], random_returns.mean(), f" random agent ({random_returns.mean():.0f})",
            color=COLORS["random"], va="center", ha="left", fontsize=9)
    ax.axhline(SUCCESS_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], SUCCESS_THRESHOLD, f" solved ({SUCCESS_THRESHOLD:.0f})",
            va="center", ha="left", fontsize=9)

    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"mean return over 10 evaluation episodes")
    ax.set_title("LunarLander-v3: learning curves, evaluated every 25k steps")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_returns(runs_dir: Path, out_path: Path) -> None:
    rng = np.random.default_rng(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    position = 0
    ticks, tick_labels = [], []

    for algo in ("random", "dqn", "ppo"):
        for run_dir in seed_run_dirs(runs_dir, algo):
            returns = load_returns(run_dir / "final_eval.csv")
            shown = returns[returns >= FINAL_RETURN_FLOOR]
            ax.scatter(position + rng.uniform(-0.25, 0.25, shown.size), shown, s=9,
                       color=COLORS[algo], alpha=0.55, linewidths=0)
            ax.plot([position - 0.32, position + 0.32], [returns.mean()] * 2, color="black", linewidth=2)
            ax.text(position, 380, f"{returns.mean():.0f}", ha="center", fontsize=8.5)
            hidden = returns.size - shown.size
            if hidden:
                ax.text(position, FINAL_RETURN_FLOOR + 30,
                        f"{hidden} below\n(min {returns.min():.0f})", ha="center", fontsize=7.5)
            ticks.append(position)
            tick_labels.append(f"{LABELS[algo]}\n{run_dir.name.replace('_', ' ')}")
            position += 1
        position += 0.6

    ax.axhline(SUCCESS_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(ticks, tick_labels, fontsize=8.5)
    ax.set_ylim(FINAL_RETURN_FLOOR, 420)
    ax.set_ylabel("episode return")
    ax.set_title("Final policies: 100 held-out episodes per run (black bar and number: mean)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"),
                        help="config whose evaluation grid is used to align seeds")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    every = load_config(args.config).evaluation.every_timesteps
    plot_learning_curves(args.runs_dir, args.out_dir / "learning_curves.png", every)
    plot_final_returns(args.runs_dir, args.out_dir / "final_returns.png")
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
