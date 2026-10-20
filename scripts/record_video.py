"""Before/after demo: four policies play the same episode side by side.

    python scripts/record_video.py --runs-dir runs --seed 0

All panels start from the same held-out episode seed (the first one of the
final evaluation, 1,000,000), which is not picked by looking at the result.
When an episode ends, its panel freezes on the last frame until the longest
one is over. Writes a GIF for the README and an MP4 at full resolution.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO

from lunarlander_rl.envs import make_env
from lunarlander_rl.evaluate import FINAL_EVAL_BASE_SEED
from lunarlander_rl.evaluation import Policy, RandomPolicy, sb3_policy

ENV_FPS = 50  # LunarLander's own rendering rate
GIF_FRAME_STEP = 2  # keep 1 frame in 2 for the GIF, played at 25 fps = real time
GIF_SCALE = 0.5


@dataclass
class Panel:
    title: str
    policy: Policy


@dataclass
class Episode:
    frames: list[np.ndarray]
    episode_return: float


def play(policy: Policy, episode_seed: int) -> Episode:
    env = make_env(render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=episode_seed)
        frames, total, done = [env.render()], 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += float(reward)
            frames.append(env.render())
            done = terminated or truncated
    finally:
        env.close()
    return Episode(frames=frames, episode_return=total)


def label(frame: np.ndarray, title: str, subtitle: str, font_size: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    draw.text((8, 8 + font_size), subtitle, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def compose(panels: list[Panel], episodes: list[Episode], step: int, scale: float) -> np.ndarray:
    """2x2 grid of the panels at a given step (finished episodes stay frozen)."""
    tiles = []
    for panel, episode in zip(panels, episodes):
        index = min(step, len(episode.frames) - 1)
        frame = episode.frames[index]
        if scale != 1.0:
            height, width = frame.shape[:2]
            frame = np.asarray(Image.fromarray(frame).resize(
                (int(width * scale), int(height * scale)), Image.Resampling.BILINEAR))
        finished = index == len(episode.frames) - 1
        subtitle = (f"return {episode.episode_return:.0f}" if finished
                    else f"step {index}")
        tiles.append(label(frame, panel.title, subtitle, font_size=max(10, int(22 * scale))))
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0, help="training seed of the runs to show")
    parser.add_argument("--episode-seed", type=int, default=FINAL_EVAL_BASE_SEED)
    parser.add_argument("--gif", type=Path, default=Path("figures/before_after.gif"))
    parser.add_argument("--mp4", type=Path, default=Path("videos/before_after.mp4"))
    args = parser.parse_args()

    dqn_dir = args.runs_dir / "dqn" / f"seed_{args.seed}"
    ppo_dir = args.runs_dir / "ppo" / f"seed_{args.seed}"
    model_paths = [dqn_dir / "checkpoints" / "step_100000.zip", dqn_dir / "final_model.zip",
                   ppo_dir / "final_model.zip"]
    missing = [str(p) for p in model_paths if not p.exists()]
    if missing:
        parser.error(f"missing models: {missing}")

    panels = [
        Panel("Random agent", RandomPolicy(seed=args.seed)),
        Panel("DQN after 100k steps", sb3_policy(DQN.load(model_paths[0], device="cpu"))),
        Panel("DQN after 1M steps", sb3_policy(DQN.load(model_paths[1], device="cpu"))),
        Panel("PPO after 1M steps", sb3_policy(PPO.load(model_paths[2], device="cpu"))),
    ]
    episodes = [play(panel.policy, args.episode_seed) for panel in panels]
    n_steps = max(len(episode.frames) for episode in episodes)
    for panel, episode in zip(panels, episodes):
        print(f"{panel.title}: {len(episode.frames) - 1} steps, return {episode.episode_return:.1f}")

    # Hold the final picture for one second so the returns can be read.
    hold = list(range(n_steps)) + [n_steps - 1] * ENV_FPS

    args.mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.mp4, fps=ENV_FPS, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for step in hold:
            writer.append_data(compose(panels, episodes, step, scale=1.0))

    args.gif.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = [compose(panels, episodes, step, scale=GIF_SCALE) for step in hold[::GIF_FRAME_STEP]]
    imageio.mimsave(args.gif, gif_frames, duration=1000 * GIF_FRAME_STEP / ENV_FPS, loop=0)
    print(f"{args.gif} ({args.gif.stat().st_size / 1e6:.1f} MB), {args.mp4} "
          f"({args.mp4.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
