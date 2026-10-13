from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.articulated import AllegroHandEnvironment, POWER_GRASP_POSE, smooth_grasp_command


def execute(env: AllegroHandEnvironment, seed: int, frames: list[Image.Image] | None = None) -> dict:
    env.reset(seed=seed, domain_randomize=True)
    renderer = None
    camera = None
    if frames is not None:
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.05]
        camera.distance = 0.55
        camera.azimuth = 140
        camera.elevation = -25
    steps = 120
    for step in range(steps):
        env.step(smooth_grasp_command(step, steps))
        if renderer is not None and step % 3 == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(Image.fromarray(renderer.render().copy()))
    if renderer is not None:
        renderer.close()
    error = float(np.sqrt(np.mean((env.data.qpos[:16] - POWER_GRASP_POSE) ** 2)))
    contacts = env.hand_object_contacts()
    return {"joint_rmse_rad": error, "hand_object_contacts": contacts, "success": error < 0.08 and contacts >= 2}


def main() -> None:
    env = AllegroHandEnvironment(ROOT / "third_party" / "wonik_allegro" / "scene_right.xml")
    trials = [execute(env, 30000 + index) for index in range(100)]
    frames: list[Image.Image] = []
    execute(env, 7, frames)
    output = ROOT / "artifacts" / "articulated"
    output.mkdir(parents=True, exist_ok=True)
    frames[0].save(output / "allegro_power_grasp.gif", save_all=True, append_images=frames[1:], duration=75, loop=0)
    report = {
        "model": "MuJoCo Menagerie Wonik Allegro Hand V3",
        "actuated_dof": env.model.nu,
        "episodes": len(trials),
        "successes": sum(int(trial["success"]) for trial in trials),
        "success_rate": float(np.mean([trial["success"] for trial in trials])),
        "mean_joint_rmse_rad": float(np.mean([trial["joint_rmse_rad"] for trial in trials])),
        "mean_hand_object_contacts": float(np.mean([trial["hand_object_contacts"] for trial in trials])),
        "criterion": "joint RMSE < 0.08 rad and >=2 hand-object contacts",
        "domain_randomization": "object mass 0.65-1.40x, friction 0.60-1.50x, object pose and joint initialization",
        "trials": trials,
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
