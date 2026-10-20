"""Environment factory shared by training, evaluation and robustness scripts.

Every script builds its environment through this module so that the
environment id and the seeding logic live in exactly one place.
"""

import gymnasium as gym

ENV_ID = "LunarLander-v3"

# LunarLander observation: x, y, vx, vy, angle, angular velocity,
# left leg contact, right leg contact.
OBSERVATION_SIZE = 8

# Discrete actions: 0 = do nothing, 1 = fire left engine,
# 2 = fire main engine, 3 = fire right engine.
N_ACTIONS = 4


def make_env(seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create a discrete LunarLander environment, seeded if a seed is given.

    Seeding goes through reset() and the action space, which is how
    Gymnasium expects it. Calling env.seed() no longer exists in Gymnasium 1.x.
    """
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
