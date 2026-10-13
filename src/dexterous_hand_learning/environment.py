from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .safety import SafetyMonitor


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    info: dict[str, float | int | str | bool]


class HandEnvironment:
    """Small MuJoCo multi-contact hand transport task.

    The model has a Cartesian palm and four independently actuated fingers. It is
    intentionally compact so the full data-to-policy workflow runs on a CPU.
    """

    def __init__(
        self,
        model_path: str | Path,
        control_dt: float = 0.02,
        episode_steps: int = 220,
        success_radius_m: float = 0.055,
        max_action_age_s: float = 0.1,
    ):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.frame_skip = max(1, round(self.control_dt / self.model.opt.timestep))
        self.episode_steps = int(episode_steps)
        self.success_radius_m = float(success_radius_m)
        self.target_xy = np.array([0.14, 0.08], dtype=np.float64)
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.action_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.action_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.safety = SafetyMonitor(self.action_low, self.action_high, max_action_age_s)
        self.step_index = 0
        self.safety_events = 0
        self._nominal_cube_mass = float(self.model.body_mass[self.cube_body_id])
        self._nominal_cube_inertia = self.model.body_inertia[self.cube_body_id].copy()
        self._nominal_cube_friction = self.model.geom_friction[self.cube_geom_id].copy()

    @property
    def action_size(self) -> int:
        return int(self.model.nu)

    @property
    def observation_size(self) -> int:
        return int(self.observe(0.0).size)

    def reset(
        self,
        seed: int | None = None,
        randomize: bool = True,
        randomization_scale: float = 1.0,
        domain_randomize: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        cube_xy = np.array([-0.12, 0.0])
        target_xy = np.array([0.14, 0.08])
        if randomize:
            span = 0.025 * float(randomization_scale)
            cube_xy += rng.uniform([-span, -span], [span, span])
            target_xy += rng.uniform([-span, -span], [span, span])
        self.target_xy = target_xy
        self.model.body_mass[self.cube_body_id] = self._nominal_cube_mass
        self.model.body_inertia[self.cube_body_id] = self._nominal_cube_inertia
        self.model.geom_friction[self.cube_geom_id] = self._nominal_cube_friction
        if domain_randomize:
            mass_scale = rng.uniform(0.65, 1.40)
            friction_scale = rng.uniform(0.60, 1.50)
            self.model.body_mass[self.cube_body_id] *= mass_scale
            self.model.body_inertia[self.cube_body_id] *= mass_scale
            self.model.geom_friction[self.cube_geom_id, 0] *= friction_scale

        # Palm joints followed by four finger joints, then cube free-joint qpos.
        self.data.qpos[:7] = np.array([cube_xy[0], cube_xy[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[7:10] = np.array([cube_xy[0], cube_xy[1], 0.026])
        self.data.qpos[10:14] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.ctrl[:] = np.array([cube_xy[0], cube_xy[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)
        self.step_index = 0
        self.safety_events = 0
        return self.observe(0.0)

    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def palm_position(self) -> np.ndarray:
        return self.data.xpos[self.palm_body_id].copy()

    def observe(self, phase: float) -> np.ndarray:
        cube_pos = self.cube_position()
        palm_pos = self.palm_position()
        return np.concatenate(
            [
                self.data.qpos[:7].copy(),
                self.data.qvel[:7].copy(),
                cube_pos,
                self.data.cvel[self.cube_body_id].copy(),
                palm_pos,
                self.target_xy.copy(),
                np.array([phase], dtype=np.float64),
            ]
        ).astype(np.float32)

    def step(self, command: np.ndarray, phase: float, action_age_s: float = 0.0) -> StepResult:
        check = self.safety.validate(command, action_age_s)
        if not check.safe:
            self.safety_events += 1
        self.data.ctrl[:] = check.command
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        cube = self.cube_position()
        distance = float(np.linalg.norm(cube[:2] - self.target_xy))
        success = distance <= self.success_radius_m and cube[2] < 0.07
        timeout = self.step_index >= self.episode_steps
        unstable = not np.all(np.isfinite(self.data.qpos))
        reward = -distance + (1.0 if success else 0.0)
        return StepResult(
            self.observe(phase),
            reward,
            bool(success or timeout or unstable),
            {
                "success": bool(success),
                "timeout": bool(timeout),
                "unstable": bool(unstable),
                "distance_m": distance,
                "safety_events": self.safety_events,
                "safety_reason": check.reason,
            },
        )
