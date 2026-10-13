from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.expert import scripted_expert


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    env = HandEnvironment(
        ROOT / "assets" / "four_finger_hand.xml",
        config.control_dt,
        config.episode_steps,
        config.success_radius_m,
        config.max_action_age_s,
    )
    observation = env.reset(seed=config.seed, randomize=False)
    start_xy = env.cube_position()[:2].copy()
    trace: list[dict[str, object]] = []
    success = False
    for step in range(config.episode_steps):
        phase = step / max(1, config.episode_steps - 1)
        action = scripted_expert(step, config.episode_steps, start_xy, env.target_xy)
        result = env.step(action, phase)
        observation = result.observation
        success = success or bool(result.info["success"])
        if step % 20 == 0 or step == config.episode_steps - 1:
            trace.append(
                {
                    "step": step,
                    "cube_m": np.round(env.cube_position(), 4).tolist(),
                    "palm_m": np.round(env.palm_position(), 4).tolist(),
                    "distance_m": round(float(result.info["distance_m"]), 4),
                    "success": bool(result.info["success"]),
                }
            )
    print(json.dumps({"ever_success": success, "final": trace[-1], "trace": trace}, indent=2))


if __name__ == "__main__":
    main()
