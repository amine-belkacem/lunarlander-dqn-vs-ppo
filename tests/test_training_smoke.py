"""End-to-end check of the training pipeline on a tiny budget.

These runs are far too short to learn anything. They only check that the
pipeline produces the expected files, on the expected evaluation grid.
"""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lunarlander_rl import evaluate
from lunarlander_rl.config import EvaluationSettings, load_config
from lunarlander_rl.train import train

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TINY_EVAL = EvaluationSettings(every_timesteps=1024, episodes=2, base_seed=10_000)


def _curve_timesteps(run_dir: Path) -> tuple[list[int], int]:
    """Distinct evaluation steps in curve.csv, and its number of rows."""
    with (run_dir / "curve.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted({int(row["timesteps"]) for row in rows}), len(rows)


@pytest.mark.slow
def test_dqn_smoke_run(tmp_path):
    config = replace(
        load_config(CONFIG_DIR / "dqn.yaml"),
        total_timesteps=2048,
        evaluation=TINY_EVAL,
        checkpoints=(1500,),
        hyperparams={**load_config(CONFIG_DIR / "dqn.yaml").hyperparams, "buffer_size": 2048},
    )
    run_dir = train(config, seed=0, out_dir=tmp_path)

    # DQN advances one step at a time: evaluations at 0, 1024 and 2048,
    # 2 episodes each -> 3 * 2 = 6 rows.
    grid, n_rows = _curve_timesteps(run_dir)
    assert grid == [0, 1024, 2048], f"ECHEC : obtenu {grid}, attendu [0, 1024, 2048]"
    assert n_rows == 6, f"ECHEC : obtenu {n_rows} lignes, attendu 6"

    assert (run_dir / "final_model.zip").exists(), "ECHEC : final_model.zip absent, attendu présent"
    assert (run_dir / "checkpoints" / "step_1500.zip").exists(), (
        "ECHEC : checkpoints/step_1500.zip absent, attendu présent"
    )
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done", f"ECHEC : obtenu {meta['status']}, attendu done"
    assert meta["actual_timesteps"] == 2048, f"ECHEC : obtenu {meta['actual_timesteps']}, attendu 2048"


@pytest.mark.slow
def test_ppo_smoke_run_and_final_evaluation(tmp_path):
    base = load_config(CONFIG_DIR / "ppo.yaml")
    config = replace(
        base,
        total_timesteps=2048,
        n_envs=2,
        evaluation=TINY_EVAL,
        checkpoints=(),
        hyperparams={**base.hyperparams, "n_steps": 64, "batch_size": 32, "n_epochs": 1},
    )
    run_dir = train(config, seed=0, out_dir=tmp_path)

    # One update = 2 envs * 64 steps = 128 steps; 2048 / 128 = 16 updates exactly,
    # so training stops at 2048. Steps go up by 2, so 1024 and 2048 are hit exactly.
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["actual_timesteps"] == 2048, f"ECHEC : obtenu {meta['actual_timesteps']}, attendu 2048"
    grid, n_rows = _curve_timesteps(run_dir)
    assert grid == [0, 1024, 2048], f"ECHEC : obtenu {grid}, attendu [0, 1024, 2048]"
    assert n_rows == 6, f"ECHEC : obtenu {n_rows} lignes, attendu 6"

    evaluate.main(["--run-dir", str(run_dir), "--episodes", "3"])
    with (run_dir / "final_eval.csv").open(encoding="utf-8") as handle:
        seeds = [int(row["episode_seed"]) for row in csv.DictReader(handle)]
    expected = [1_000_000, 1_000_001, 1_000_002]
    assert seeds == expected, f"ECHEC : obtenu {seeds}, attendu {expected}"
