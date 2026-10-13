from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .articulated import AllegroHandEnvironment


NOMINAL_GRASP_POSE = np.array([0.0, 0.75, 0.75, 0.75] * 3 + [1.0, 0.50, 0.50, 0.50], dtype=np.float64)


@dataclass(frozen=True)
class TactileState:
    contact_count: int
    normal_force_n: float
    object_speed_mps: float
    slip_detected: bool


def estimate_tactile_state(env: AllegroHandEnvironment) -> TactileState:
    contact_count = 0
    normal_force = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        body1 = int(env.model.geom_bodyid[contact.geom1])
        body2 = int(env.model.geom_bodyid[contact.geom2])
        if env.object_body_id in (body1, body2) and 0 not in (body1, body2):
            mujoco.mj_contactForce(env.model, env.data, index, wrench)
            normal_force += max(0.0, float(wrench[0]))
            contact_count += 1
    speed = float(np.linalg.norm(env.data.cvel[env.object_body_id, 3:]))
    slip = contact_count < 2 or speed > 0.035 or normal_force < 0.08
    return TactileState(contact_count, normal_force, speed, slip)


class TactileRecoveryController:
    """Adds bounded grip closure when contact or motion indicates object slip."""

    def __init__(self, env: AllegroHandEnvironment):
        self.low = env.low
        self.high = env.high
        self.recovery_level = 0.0

    def update(self, tactile: TactileState) -> np.ndarray:
        if tactile.slip_detected:
            self.recovery_level = min(1.0, self.recovery_level + 0.10)
        else:
            self.recovery_level = max(0.0, self.recovery_level - 0.02)
        delta = np.array([0.0, 0.22, 0.20, 0.16] * 3 + [0.12, 0.18, 0.18, 0.14])
        return np.clip(NOMINAL_GRASP_POSE + self.recovery_level * delta, self.low, self.high)
