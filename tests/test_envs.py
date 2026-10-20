import numpy as np

from lunarlander_rl.envs import N_ACTIONS, OBSERVATION_SIZE, make_env


def test_observation_and_action_spaces():
    env = make_env(seed=0)
    obs, _ = env.reset(seed=0)
    env.close()

    assert obs.shape == (OBSERVATION_SIZE,), (
        f"ECHEC : obtenu {obs.shape}, attendu {(OBSERVATION_SIZE,)}"
    )
    assert env.action_space.n == N_ACTIONS, (
        f"ECHEC : obtenu {env.action_space.n}, attendu {N_ACTIONS}"
    )


def _rollout(seed: int, n_steps: int) -> np.ndarray:
    """Play n_steps random actions and return the stacked observations."""
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    observations = [obs]
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        observations.append(obs)
        if terminated or truncated:
            break
    env.close()
    return np.array(observations)


def test_same_seed_gives_identical_trajectory():
    first = _rollout(seed=42, n_steps=50)
    second = _rollout(seed=42, n_steps=50)

    assert first.shape == second.shape, (
        f"ECHEC : obtenu {second.shape}, attendu {first.shape}"
    )
    max_gap = float(np.max(np.abs(first - second)))
    assert max_gap == 0.0, f"ECHEC : obtenu un écart max de {max_gap}, attendu 0.0"


def test_different_seeds_give_different_starts():
    first = _rollout(seed=1, n_steps=0)
    second = _rollout(seed=2, n_steps=0)

    assert not np.array_equal(first, second), (
        f"ECHEC : obtenu deux états initiaux identiques {first[0]}, attendu des états différents"
    )
