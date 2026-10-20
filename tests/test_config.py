from pathlib import Path

import pytest

from lunarlander_rl.config import load_config, parse_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _valid_raw() -> dict:
    return {
        "algo": "ppo",
        "total_timesteps": 1000,
        "n_envs": 2,
        "hyperparams": {"n_steps": 64},
        "evaluation": {"every_timesteps": 500, "episodes": 2, "base_seed": 0},
    }


def test_dqn_and_ppo_share_budget_and_evaluation_protocol():
    dqn = load_config(CONFIG_DIR / "dqn.yaml")
    ppo = load_config(CONFIG_DIR / "ppo.yaml")

    assert dqn.total_timesteps == ppo.total_timesteps, (
        f"ECHEC : obtenu {dqn.total_timesteps} pas pour DQN, attendu {ppo.total_timesteps} comme PPO"
    )
    assert dqn.evaluation == ppo.evaluation, (
        f"ECHEC : obtenu {dqn.evaluation} pour DQN, attendu {ppo.evaluation} comme PPO"
    )
    assert dqn.checkpoints == ppo.checkpoints, (
        f"ECHEC : obtenu {dqn.checkpoints} pour DQN, attendu {ppo.checkpoints} comme PPO"
    )


def test_dqn_epsilon_schedule_matches_the_tuned_zoo_schedule():
    # Zoo: exploration_fraction 0.12 over 1e5 steps -> 0.12 * 100_000 = 12_000 steps.
    # Here: 0.012 over 1e6 steps -> 0.012 * 1_000_000 = 12_000 steps.
    dqn = load_config(CONFIG_DIR / "dqn.yaml")
    decay_steps = dqn.hyperparams["exploration_fraction"] * dqn.total_timesteps

    assert decay_steps == pytest.approx(12_000), (
        f"ECHEC : obtenu {decay_steps} pas de décroissance d'epsilon, attendu 12000"
    )


def test_learning_rate_is_parsed_as_a_float():
    # PyYAML reads "1e-4" (no dot) as a string; "6.3e-4" must come out as a float.
    lr = load_config(CONFIG_DIR / "dqn.yaml").hyperparams["learning_rate"]

    assert isinstance(lr, float) and lr == pytest.approx(6.3e-4), (
        f"ECHEC : obtenu {lr!r} ({type(lr).__name__}), attendu le float 0.00063"
    )


def test_missing_key_is_reported():
    raw = _valid_raw()
    del raw["n_envs"]

    with pytest.raises(ValueError, match="n_envs"):
        parse_config(raw)


def test_unknown_algorithm_is_rejected():
    raw = _valid_raw()
    raw["algo"] = "a2c"

    with pytest.raises(ValueError, match="unknown algo"):
        parse_config(raw)


@pytest.mark.parametrize("bad_value", [0, -5, True, 1000.0, "1000"])
def test_total_timesteps_must_be_a_positive_integer(bad_value):
    raw = _valid_raw()
    raw["total_timesteps"] = bad_value

    with pytest.raises(ValueError, match="total_timesteps"):
        parse_config(raw)


def test_checkpoints_are_sorted():
    raw = _valid_raw()
    raw["checkpoints"] = [300, 100, 200]

    checkpoints = parse_config(raw).checkpoints
    assert checkpoints == (100, 200, 300), f"ECHEC : obtenu {checkpoints}, attendu (100, 200, 300)"
