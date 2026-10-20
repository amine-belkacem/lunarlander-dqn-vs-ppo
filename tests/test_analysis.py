import numpy as np
import pytest

from lunarlander_rl.analysis import (
    Curve,
    first_step_reaching,
    load_curve,
    mean_over_seeds,
    snap_to_grid,
)


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


def test_load_curve_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    _write_curve(path, [])
    with pytest.raises(ValueError):
        load_curve(path)
