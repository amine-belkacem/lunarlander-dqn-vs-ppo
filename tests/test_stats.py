import numpy as np
import pytest

from lunarlander_rl.stats import bootstrap_ci, iqm, probability_of_improvement


@pytest.mark.parametrize("values, expected", [
    # n = 8: int(0.25 * 8) = 2 cut per side -> mean(3, 4, 5, 6) = 18 / 4 = 4.5
    ([1, 2, 3, 4, 5, 6, 7, 8], 4.5),
    # Same with one catastrophic run: it is cut, the IQM does not move.
    ([1, 2, 3, 4, 5, 6, 7, 1000], 4.5),
    # n = 10: 2 cut per side -> mean(3..8) = 33 / 6 = 5.5
    ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 5.5),
    # n = 3: int(0.75) = 0, nothing is cut -> mean = 2
    ([1, 2, 3], 2.0),
])
def test_iqm_hand_computed(values, expected):
    result = iqm(np.array(values))
    assert result == pytest.approx(expected), f"ECHEC : obtenu {result}, attendu {expected}"


def test_iqm_rejects_empty_input():
    with pytest.raises(ValueError):
        iqm(np.array([]))


def test_bootstrap_of_two_values_hand_computed():
    # Resampling [0, 1] twice with replacement gives a mean of 0 (p = 1/4),
    # 0.5 (p = 1/2) or 1 (p = 1/4). The 2.5th percentile is therefore 0 and
    # the 97.5th is 1; the point estimate is the mean of [0, 1] = 0.5.
    est = bootstrap_ci(np.array([0.0, 1.0]), statistic=np.mean, n_bootstrap=20_000)
    assert (est.point, est.low, est.high) == (0.5, 0.0, 1.0), (
        f"ECHEC : obtenu ({est.point}, {est.low}, {est.high}), attendu (0.5, 0.0, 1.0)"
    )


def test_bootstrap_of_constant_runs_has_zero_width():
    est = bootstrap_ci(np.full(10, 250.0))
    assert (est.low, est.high) == (250.0, 250.0), (
        f"ECHEC : obtenu [{est.low}, {est.high}], attendu [250.0, 250.0]"
    )


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    values = np.array([-731.8, -544.1, -784.1, 120.0, 230.5, 15.0, -90.0])
    first = bootstrap_ci(values, n_bootstrap=5_000, seed=3)
    second = bootstrap_ci(values, n_bootstrap=5_000, seed=3)
    assert first == second, f"ECHEC : obtenu {first} puis {second}, attendu deux fois le même"
    assert first.low <= first.point <= first.high, (
        f"ECHEC : obtenu {first}, attendu low <= point <= high"
    )


@pytest.mark.parametrize("x, y, expected", [
    # Every x beats every y -> 1
    ([3, 4], [1, 2], 1.0),
    # Pairs (1,1)=1/2, (1,2)=0, (2,1)=1, (2,2)=1/2 -> 2 / 4 = 0.5
    ([1, 2], [1, 2], 0.5),
    # x=5 beats 1 and 2, loses to 9 -> 2 / 3
    ([5], [1, 2, 9], 2 / 3),
])
def test_probability_of_improvement_hand_computed(x, y, expected):
    result = probability_of_improvement(np.array(x), np.array(y))
    assert result == pytest.approx(expected), f"ECHEC : obtenu {result}, attendu {expected}"
