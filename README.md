# DQN vs PPO on LunarLander

![tests](https://github.com/amine-belkacem/lunarlander-dqn-vs-ppo/actions/workflows/tests.yml/badge.svg)

> Work in progress. This README describes what I am building; results will only appear here once they are measured.

I am comparing two reinforcement learning algorithms, DQN and PPO, on Gymnasium's `LunarLander-v3`, where an agent has to land a module between two flags using its engines.

Most comparisons of this kind you find online rely on one or two training runs. In RL that is not enough: the same algorithm can solve the task with one random seed and fail with another. So the plan here is to treat it like a measurement problem.

## What the project will answer

1. **Performance.** With the same budget of 1M environment steps, which algorithm lands better and learns faster? 10 seeds per algorithm, results reported as IQM with 95% bootstrap confidence intervals.
2. **Robustness.** Agents are trained in nominal conditions, then tested under wind, turbulence, sensor noise, action delay and a failed side engine.
3. **Deployment.** The trained policies are exported to ONNX, quantized to int8 and run on a Raspberry Pi 5, where I measure latency and check what quantization costs in landing success.

## Roadmap

| Milestone | Content | Status |
|---|---|---|
| J0 | Project setup, tests, CI | done |
| J1 | Training and evaluation pipeline, first 3 seeds, before/after video | planned |
| J2 | 10 seeds per algorithm, statistics with confidence intervals | planned |
| J3 | Robustness map under perturbations | planned |
| J4 | ONNX export and int8 quantization | planned |
| J5 | Closed-loop inference on Raspberry Pi 5 | planned |

## Setup

Python 3.10 to 3.12 (PyTorch has no wheels for newer versions yet).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
pytest -v
```

## Dependencies and licenses

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (MIT): LunarLander environment
- [Box2D](https://github.com/pybox2d/pybox2d) (zlib): physics engine used by LunarLander
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT): DQN and PPO implementations
- [PyTorch](https://github.com/pytorch/pytorch) (BSD-style)

## License

MIT
