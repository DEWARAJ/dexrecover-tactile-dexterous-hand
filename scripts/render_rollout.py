from __future__ import annotations

import sys
from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.policy import NormalizedPolicy


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    env = HandEnvironment(
        ROOT / "assets" / "four_finger_hand.xml",
        config.control_dt,
        config.episode_steps,
        config.success_radius_m,
        config.max_action_age_s,
    )
    policy = NormalizedPolicy.load(ROOT / "artifacts" / "reference" / "behavior_cloning.pt")
    observation = env.reset(seed=config.seed, randomize=False)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.02, 0.04]
    camera.distance = 0.72
    camera.azimuth = 135
    camera.elevation = -35
    frames: list[Image.Image] = []
    for step in range(config.episode_steps):
        phase = step / max(1, config.episode_steps - 1)
        result = env.step(policy(observation), phase)
        observation = result.observation
        if step % 4 == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(Image.fromarray(renderer.render().copy()))
    renderer.close()
    output = ROOT / "artifacts" / "reference" / "policy_rollout.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(output)


if __name__ == "__main__":
    main()
