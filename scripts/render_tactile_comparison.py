from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.articulated import AllegroHandEnvironment, OPEN_POSE
from dexterous_hand_learning.tactile import NOMINAL_GRASP_POSE, TactileRecoveryController, estimate_tactile_state


def rollout(recovery: bool, seed: int = 40000) -> list[Image.Image]:
    env = AllegroHandEnvironment(ROOT / "third_party" / "wonik_allegro" / "scene_right.xml")
    env.reset(seed=seed, domain_randomize=True)
    renderer = mujoco.Renderer(env.model, height=420, width=560)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.045]
    camera.distance = 0.52
    camera.azimuth = 140
    camera.elevation = -25
    controller = TactileRecoveryController(env)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    force_n = rng.uniform(0.20, 0.45)
    frames = []
    for step in range(120 + 75):
        if step < 120:
            phase = step / 119
            blend = phase * phase * (3.0 - 2.0 * phase)
            command = (1.0 - blend) * OPEN_POSE + blend * NOMINAL_GRASP_POSE
            stage = "Closing"
        else:
            disturbance_step = step - 120
            env.data.xfrc_applied[env.object_body_id] = 0.0
            if disturbance_step < 25:
                env.data.xfrc_applied[env.object_body_id, :2] = force_n * direction
                stage = "Disturbance"
            else:
                stage = "Recovery"
            tactile = estimate_tactile_state(env)
            command = controller.update(tactile) if recovery else NOMINAL_GRASP_POSE
        env.step(command)
        if step % 4 == 0:
            tactile = estimate_tactile_state(env)
            renderer.update_scene(env.data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            label = "TACTILE RECOVERY" if recovery else "FIXED GRASP"
            draw.rectangle((8, 8, 245, 60), fill=(8, 20, 30, 210))
            draw.text((16, 14), label, fill="white")
            draw.text((16, 34), f"{stage} | contacts={tactile.contact_count} | slip={tactile.slip_detected}", fill="white")
            frames.append(frame)
    renderer.close()
    return frames


def main() -> None:
    fixed = rollout(False)
    adaptive = rollout(True)
    combined = []
    for left, right in zip(fixed, adaptive):
        canvas = Image.new("RGB", (left.width + right.width, left.height), "black")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        combined.append(canvas)
    output = ROOT / "artifacts" / "tactile_recovery" / "fixed_vs_tactile_recovery.gif"
    combined[0].save(output, save_all=True, append_images=combined[1:], duration=80, loop=0)
    print(output)


if __name__ == "__main__":
    main()
