import numpy as np
import pytest

from pathlib import Path

from lunarlander_rl.analysis import (
    Curve,
    first_step_reaching,
    is_collapsed,
    last_points_mean,
    load_curve,
    mean_over_seeds,
    snap_to_grid,
)

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _write_curve(path, rows):
    lines = ["timesteps,episode_seed,return,length"]
    lines += [f"{t},{seed},{ret},100" for t, seed, ret in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_curve_averages_episodes_of_each_point(tmp_path):
    # Step 0: (-100 + -300) / 2 = -200.  Step 25000: (150 + 250) / 2 = 200.
    path = tmp_path / "curve.csv"
    _write_curve(path, [(0, 10000, -100), (0, 10001, -300),
                        (25000, 10000, 150), (25000, 10001, 250)])
    curve = load_curve(path)

    assert curve.timesteps.tolist() == [0, 25000], (
        f"ECHEC : obtenu {curve.timesteps.tolist()}, attendu [0, 25000]"
    )
    assert curve.mean_returns.tolist() == [-200.0, 200.0], (
        f"ECHEC : obtenu {curve.mean_returns.tolist()}, attendu [-200.0, 200.0]"
    )


def test_first_step_reaching_threshold():
    curve = Curve(np.array([0, 25000, 50000, 75000]), np.array([-300.0, 150.0, 200.0, 180.0]))
    # 200.0 >= 200 at 50000 (inclusive); the later drop to 180 does not matter.
    step = first_step_reaching(curve, 200.0)
    assert step == 50000, f"ECHEC : obtenu {step}, attendu 50000"


def test_first_step_reaching_returns_none_when_never_reached():
    curve = Curve(np.array([0, 25000]), np.array([-300.0, 199.9]))
    step = first_step_reaching(curve, 200.0)
    assert step is None, f"ECHEC : obtenu {step}, attendu None"


def test_snap_to_grid_handles_ppo_offsets():
    # 25008 / 25000 = 1.0003 -> 1 -> 25000 ; 49984 / 25000 = 1.99936 -> 2 -> 50000
    snapped = snap_to_grid(np.array([0, 25008, 49984, 1000000]), 25000).tolist()
    assert snapped == [0, 25000, 50000, 1000000], (
        f"ECHEC : obtenu {snapped}, attendu [0, 25000, 50000, 1000000]"
    )


def test_mean_over_seeds_keeps_only_common_points():
    a = Curve(np.array([0, 25000, 50000]), np.array([-100.0, 0.0, 100.0]))
    b = Curve(np.array([0, 25008]), np.array([-300.0, 200.0]))
    # Common grid points: 0 and 25000. Means: (-100 - 300) / 2 = -200 ; (0 + 200) / 2 = 100
    mean = mean_over_seeds([a, b], every=25000)

    assert mean.timesteps.tolist() == [0, 25000], (
        f"ECHEC : obtenu {mean.timesteps.tolist()}, attendu [0, 25000]"
    )
    assert mean.mean_returns.tolist() == [-200.0, 100.0], (
        f"ECHEC : obtenu {mean.mean_returns.tolist()}, attendu [-200.0, 100.0]"
    )


def _curve(values):
    return Curve(np.arange(len(values)) * 25000, np.array(values, dtype=float))


def test_last_points_mean():
    # Last 5 of [9, 1, 2, 3, 4, 5]: (1 + 2 + 3 + 4 + 5) / 5 = 3
    mean = last_points_mean(_curve([9, 1, 2, 3, 4, 5]), 5)
    assert mean == 3.0, f"ECHEC : obtenu {mean}, attendu 3.0"


def test_collapse_needs_to_reach_then_end_below_zero():
    # Reached 250, last 5 = (-10 - 20 - 30 - 40 - 50) / 5 = -30 < 0 -> collapsed
    assert is_collapsed(_curve([-300, 250, -10, -20, -30, -40, -50]), 200), (
        "ECHEC : obtenu non effondré, attendu effondré"
    )
    # Never reached 200 (max 150): bad, but not a collapse by definition
    assert not is_collapsed(_curve([-300, 150, -10, -20, -30, -40, -50]), 200), (
        "ECHEC : obtenu effondré, attendu non effondré (200 jamais atteint)"
    )
    # Reached, last 5 = (10 + 0 + -5 + 5 + -10) / 5 = 0, not < 0 -> not collapsed
    assert not is_collapsed(_curve([250, 10, 0, -5, 5, -10]), 200), (
        "ECHEC : obtenu effondré, attendu non effondré (moyenne finale = 0)"
    )


@pytest.mark.parametrize("seed, expected_mean", [(0, -848), (1, -391), (2, -542)])
def test_preregistration_numbers_match_the_v01_runs(seed, expected_mean):
    # docs/preregistration_j2.md quotes these last-5 means for the v0.1 DQN runs.
    curve = load_curve(RUNS_DIR / "dqn" / f"seed_{seed}" / "curve.csv")
    mean = round(last_points_mean(curve, 5))
    assert mean == expected_mean, f"ECHEC : obtenu {mean}, attendu {expected_mean}"
    assert is_collapsed(curve, 200), f"ECHEC : seed {seed} non effondrée, attendu effondrée"


def test_load_curve_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    _write_curve(path, [])
    with pytest.raises(ValueError):
        load_curve(path)
