"""Train and evaluate several (config, seed) runs, a few at a time.

    python -m lunarlander_rl.launcher dqn:3-8
    python -m lunarlander_rl.launcher dqn:9 dqn_big_buffer:0-4 --max-parallel 3
    python -m lunarlander_rl.launcher ppo:3-9

Works the same on my Windows laptop and on a Linux machine (Kaggle).

Choices, all from measurements on my laptop (Ryzen 3 7320U, 4 cores):
- 1 torch thread per run: twice as fast as 4 threads for these small nets.
- Runs whose meta.json says "done" are skipped, so a stopped launcher can be
  started again. A run cut in the middle restarts from zero.
- DQN configs go first: they take about 10x longer than PPO, so starting
  them early shortens the total time.
- Starts are staggered: two processes importing torch at the same instant
  failed once on this machine (it turned out the PC was out of memory, but
  a few seconds of delay cost nothing).
- On Windows the PC is kept awake while runs are in progress.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class Job:
    config: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.config}_seed{self.seed}"


def parse_seeds(text: str) -> list[int]:
    """'3-6' -> [3, 4, 5, 6] ; '1,4,7' -> [1, 4, 7] ; '0-1,5' -> [0, 1, 5]"""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty seed in {text!r}")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"seed range {part!r} goes backwards")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"seeds must be >= 0 in {text!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {text!r}")
    return seeds


def parse_job_spec(spec: str) -> list[Job]:
    """'dqn:3-5' -> [Job('dqn', 3), Job('dqn', 4), Job('dqn', 5)]"""
    if spec.count(":") != 1:
        raise ValueError(f"expected <config>:<seeds>, got {spec!r}")
    config, seeds_text = spec.split(":")
    if not config:
        raise ValueError(f"missing config name in {spec!r}")
    return [Job(config, seed) for seed in parse_seeds(seeds_text)]


def is_done(runs_dir: Path, job: Job) -> bool:
    meta = runs_dir / job.config / f"seed_{job.seed}" / "meta.json"
    if not meta.exists():
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("status") == "done"


def build_queue(jobs: list[Job], runs_dir: Path) -> list[Job]:
    """Drop finished runs and duplicates, put DQN configs first (stable order)."""
    unique = list(dict.fromkeys(jobs))
    pending = [job for job in unique if not is_done(runs_dir, job)]
    return sorted(pending, key=lambda job: 0 if job.config.startswith("dqn") else 1)


def _keep_awake(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if enabled else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def _log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def run_jobs(queue: list[Job], runs_dir: Path, max_parallel: int, stagger_seconds: float,
             total_timesteps: int | None) -> list[Job]:
    """Run the queue with at most max_parallel processes. Returns failed jobs."""
    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[Job, subprocess.Popen, object]] = []
    failed: list[Job] = []
    queue = list(queue)

    while queue or running:
        for job, process, log_file in list(running):
            if process.poll() is not None:
                log_file.close()
                running.remove((job, process, log_file))
                if process.returncode == 0:
                    _log(f"done {job.name}")
                else:
                    failed.append(job)
                    _log(f"FAILED {job.name} (exit {process.returncode}), see {log_dir / job.name}.log")

        if queue and len(running) < max_parallel:
            job = queue.pop(0)
            command = [sys.executable, "-W", "ignore", "-m", "lunarlander_rl.train",
                       "--config", str(CONFIG_DIR / f"{job.config}.yaml"),
                       "--seed", str(job.seed), "--out-dir", str(runs_dir), "--threads", "1"]
            if total_timesteps is not None:
                command += ["--total-timesteps", str(total_timesteps)]
            log_file = open(log_dir / f"{job.name}.log", "w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                       cwd=PROJECT_ROOT)
            running.append((job, process, log_file))
            _log(f"started {job.name} ({len(queue)} waiting)")
            time.sleep(stagger_seconds)
        else:
            time.sleep(5)
    return failed


def evaluate_finished(jobs: list[Job], runs_dir: Path) -> None:
    for job in jobs:
        run_dir = runs_dir / job.config / f"seed_{job.seed}"
        if is_done(runs_dir, job) and not (run_dir / "final_eval.csv").exists():
            subprocess.run([sys.executable, "-W", "ignore", "-m", "lunarlander_rl.evaluate",
                            "--run-dir", str(run_dir)], check=True, cwd=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs", nargs="+", help="<config>:<seeds>, e.g. dqn:3-8 or ppo:0,2,4")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--stagger-seconds", type=float, default=15.0)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="override the budget (speed tests only); skips final evaluation")
    args = parser.parse_args(argv)

    try:
        jobs = [job for spec in args.jobs for job in parse_job_spec(spec)]
    except ValueError as error:
        parser.error(str(error))
    missing = sorted({job.config for job in jobs if not (CONFIG_DIR / f"{job.config}.yaml").exists()})
    if missing:
        parser.error(f"no config file for {missing} in {CONFIG_DIR}")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    queue = build_queue(jobs, args.runs_dir)
    skipped = len(set(jobs)) - len(queue)
    _log(f"{len(queue)} runs to do, {skipped} already done, {args.max_parallel} in parallel")

    _keep_awake(True)
    try:
        failed = run_jobs(queue, args.runs_dir, args.max_parallel, args.stagger_seconds,
                          args.total_timesteps)
        if args.total_timesteps is None:
            evaluate_finished(jobs, args.runs_dir)
    finally:
        _keep_awake(False)

    if failed:
        _log(f"failed runs: {', '.join(job.name for job in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
