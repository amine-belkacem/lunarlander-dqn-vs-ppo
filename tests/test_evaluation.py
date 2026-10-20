import numpy as np
import pytest

from lunarlander_rl.envs import N_ACTIONS
from lunarlander_rl.evaluation import RandomPolicy, run_episodes, success_rate


def do_nothing(obs):
    return 0


def test_success_rate_threshold_is_inclusive():
    # 250 >= 200 yes, 199.9 no, 200 yes (inclusive), -100 no -> 2 / 4 = 0.5
    rate = success_rate(np.array([250.0, 199.9, 200.0, -100.0]))
    assert rate == 0.5, f"ECHEC : obtenu {rate}, attendu 0.5"


def test_success_rate_rejects_empty_input():
    with pytest.raises(ValueError):
        success_rate(np.array([]))


def test_episode_seeds_are_consecutive_from_base_seed():
    results = run_episodes(do_nothing, n_episodes=3, base_seed=4)
    assert results.seeds.tolist() == [4, 5, 6], (
        f"ECHEC : obtenu {results.seeds.tolist()}, attendu [4, 5, 6]"
    )


def test_episode_i_depends_only_on_its_own_seed():
    # Episode with seed 5 is episode 1 of a run starting at 4, and episode 0
    # of a run starting at 5. Both must give exactly the same return.
    from_4 = run_episodes(do_nothing, n_episodes=3, base_seed=4)
    from_5 = run_episodes(do_nothing, n_episodes=2, base_seed=5)

    assert from_5.returns.tolist() == from_4.returns[1:].tolist(), (
        f"ECHEC : obtenu {from_5.returns.tolist()}, attendu {from_4.returns[1:].tolist()}"
    )


def test_episodes_end_within_the_time_limit():
    # LunarLander-v3 is wrapped in a 1000-step TimeLimit.
    results = run_episodes(do_nothing, n_episodes=3, base_seed=0)
    longest = int(results.lengths.max())
    assert 1 <= longest <= 1000, f"ECHEC : obtenu une longueur max de {longest}, attendu entre 1 et 1000"


def test_random_policy_is_reproducible_and_in_range():
    obs = np.zeros(8)
    policy_a, policy_b = RandomPolicy(seed=3), RandomPolicy(seed=3)
    actions_a = [policy_a(obs) for _ in range(200)]
    actions_b = [policy_b(obs) for _ in range(200)]

    assert actions_a == actions_b, "ECHEC : obtenu deux séquences différentes, attendu deux séquences identiques"
    # With 200 draws, the chance of never seeing one of the 4 actions is
    # 4 * (3/4)^200, about 1e-25: every action must show up.
    assert set(actions_a) == set(range(N_ACTIONS)), (
        f"ECHEC : obtenu les actions {sorted(set(actions_a))}, attendu {list(range(N_ACTIONS))}"
    )
