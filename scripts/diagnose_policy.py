from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.policy import NormalizedPolicy


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    env = HandEnvironment(ROOT / "assets" / "four_finger_hand.xml", config.control_dt, config.episode_steps)
    policy = NormalizedPolicy.load(ROOT / "artifacts" / "reference" / "behavior_cloning.pt")
    observation = env.reset(seed=10007, randomize=True)
    trace = []
    for step in range(config.episode_steps):
        phase = step / max(1, config.episode_steps - 1)
        command = policy(observation)
        result = env.step(command, phase)
        observation = result.observation
        if step % 20 == 0 or step == config.episode_steps - 1:
            trace.append({
                "step": step,
                "phase": round(phase, 3),
                "command": np.round(command, 4).tolist(),
                "cube": np.round(env.cube_position(), 4).tolist(),
                "palm": np.round(env.palm_position(), 4).tolist(),
                "distance": round(float(result.info["distance_m"]), 4),
            })
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
