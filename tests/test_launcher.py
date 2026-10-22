import json

import pytest

from lunarlander_rl.launcher import Job, build_queue, parse_job_spec, parse_seeds


@pytest.mark.parametrize("text, expected", [
    ("3-6", [3, 4, 5, 6]),
    ("1,4,7", [1, 4, 7]),
    ("0-1,5", [0, 1, 5]),
    ("9", [9]),
])
def test_parse_seeds(text, expected):
    seeds = parse_seeds(text)
    assert seeds == expected, f"ECHEC : obtenu {seeds}, attendu {expected}"


@pytest.mark.parametrize("bad", ["", "5-3", "1,,2", "1,1", "-2", "a"])
def test_parse_seeds_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_seeds(bad)


def test_parse_job_spec():
    jobs = parse_job_spec("dqn_big_buffer:0-2")
    expected = [Job("dqn_big_buffer", 0), Job("dqn_big_buffer", 1), Job("dqn_big_buffer", 2)]
    assert jobs == expected, f"ECHEC : obtenu {jobs}, attendu {expected}"


@pytest.mark.parametrize("bad", ["dqn", "dqn:1:2", ":3"])
def test_parse_job_spec_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_job_spec(bad)


def _mark_done(runs_dir, config, seed, status="done"):
    run_dir = runs_dir / config / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"status": status}), encoding="utf-8")


def test_queue_skips_done_runs_and_puts_dqn_first(tmp_path):
    _mark_done(tmp_path, "dqn", 3)                       # finished -> skipped
    _mark_done(tmp_path, "dqn", 4, status="running")     # interrupted -> redone
    jobs = [Job("ppo", 3), Job("dqn", 3), Job("dqn", 4), Job("dqn_big_buffer", 0), Job("ppo", 3)]

    queue = build_queue(jobs, tmp_path)
    # dqn 3 dropped (done), duplicate ppo 3 dropped, DQN configs before PPO,
    # original order kept inside each group.
    expected = [Job("dqn", 4), Job("dqn_big_buffer", 0), Job("ppo", 3)]
    assert queue == expected, f"ECHEC : obtenu {queue}, attendu {expected}"
