from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.teleoperation import TeleoperationController


def read_key() -> str:
    if sys.platform == "win32":
        import msvcrt

        return msvcrt.getwch().lower()
    return input("command> ").strip().lower()[:1]


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    env = HandEnvironment(ROOT / "assets" / "four_finger_hand.xml", config.control_dt, config.episode_steps)
    observation = env.reset(seed=config.seed, randomize=True)
    controller = TeleoperationController(env.action_low, env.action_high, env.data.ctrl.copy())
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    print("w/s: x  a/d: y  r/f: z  g/o: close/open  n: neutral  q: save and quit")
    step = 0
    while step < config.episode_steps:
        key = read_key()
        if key == "q":
            break
        command = controller.update(key)
        phase = step / max(1, config.episode_steps - 1)
        observations.append(observation.copy())
        actions.append(command.astype(np.float32))
        result = env.step(command, phase)
        observation = result.observation
        step += 1
        cube = env.cube_position()
        print(f"step={step:03d} key={key!r} cube={cube.round(3)} target={env.target_xy.round(3)}")

    output = ROOT / "artifacts" / "teleoperation" / "demonstration.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if observations:
        np.savez_compressed(output, observations=np.stack(observations), actions=np.stack(actions))
    metadata = {"steps": len(observations), "target_xy_m": env.target_xy.tolist(), "output": str(output)}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
