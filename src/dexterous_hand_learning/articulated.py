from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .safety import SafetyMonitor


OPEN_POSE = np.array([0.0, 0.0, 0.0, 0.0] * 3 + [0.30, 0.0, 0.0, 0.0], dtype=np.float64)
POWER_GRASP_POSE = np.array([0.0, 1.0, 1.0, 1.0] * 3 + [1.10, 0.70, 0.70, 0.70], dtype=np.float64)


class AllegroHandEnvironment:
    """Thin environment around the official 16-DOF MuJoCo Menagerie Allegro Hand."""

    def __init__(self, scene_path: str | Path, control_dt: float = 0.02):
        self.scene_path = Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.frame_skip = max(1, round(self.control_dt / self.model.opt.timestep))
        self.low = self.model.actuator_ctrlrange[:, 0].copy()
        self.high = self.model.actuator_ctrlrange[:, 1].copy()
        self.safety = SafetyMonitor(self.low, self.high)
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        object_geoms = np.flatnonzero(self.model.geom_bodyid == self.object_body_id)
        self.object_geom_id = int(object_geoms[0])
        self._nominal_mass = float(self.model.body_mass[self.object_body_id])
        self._nominal_inertia = self.model.body_inertia[self.object_body_id].copy()
        self._nominal_friction = self.model.geom_friction[self.object_geom_id].copy()

    def reset(self, seed: int = 0, domain_randomize: bool = True) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self.object_body_id] = self._nominal_mass
        self.model.body_inertia[self.object_body_id] = self._nominal_inertia
        self.model.geom_friction[self.object_geom_id] = self._nominal_friction
        if domain_randomize:
            mass_scale = rng.uniform(0.65, 1.40)
            self.model.body_mass[self.object_body_id] *= mass_scale
            self.model.body_inertia[self.object_body_id] *= mass_scale
            self.model.geom_friction[self.object_geom_id, 0] *= rng.uniform(0.60, 1.50)
        initial = np.clip(OPEN_POSE + rng.normal(0.0, 0.015, 16), self.low, self.high)
        self.data.qpos[:16] = initial
        self.data.qpos[16:19] = [rng.uniform(-0.005, 0.005), rng.uniform(-0.005, 0.005), 0.10]
        self.data.qpos[19:23] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = initial
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[:16],
            self.data.qvel[:16],
            self.data.xpos[self.object_body_id],
        ]).astype(np.float32)

    def step(self, command: np.ndarray) -> np.ndarray:
        checked = self.safety.validate(command)
        self.data.ctrl[:] = checked.command
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def hand_object_contacts(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if self.object_body_id in (body1, body2) and 0 not in (body1, body2):
                count += 1
        return count


def smooth_grasp_command(step: int, steps: int) -> np.ndarray:
    phase = np.clip(step / max(1, steps - 1), 0.0, 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    return (1.0 - blend) * OPEN_POSE + blend * POWER_GRASP_POSE
