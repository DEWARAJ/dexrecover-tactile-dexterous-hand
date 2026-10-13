from __future__ import annotations

import numpy as np


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def scripted_expert(
    step: int,
    horizon: int,
    cube_xy: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    """Time-indexed grasp, lift, transport, lower, and release command."""
    phase = step / max(1, horizon - 1)
    start_xy = np.asarray(cube_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    open_fingers = 0.0
    closed_fingers = 0.043

    if phase < 0.18:  # settle over object
        xy = start_xy
        z = 0.0
        grip = open_fingers
    elif phase < 0.32:  # close
        u = _smoothstep((phase - 0.18) / 0.14)
        xy = start_xy
        z = 0.0
        grip = open_fingers + u * (closed_fingers - open_fingers)
    elif phase < 0.46:  # lift
        u = _smoothstep((phase - 0.32) / 0.14)
        xy = start_xy
        z = 0.055 * u
        grip = closed_fingers
    elif phase < 0.72:  # transport
        u = _smoothstep((phase - 0.46) / 0.26)
        xy = (1.0 - u) * start_xy + u * target_xy
        z = 0.055
        grip = closed_fingers
    elif phase < 0.86:  # lower
        u = _smoothstep((phase - 0.72) / 0.14)
        xy = target_xy
        z = 0.055 * (1.0 - u)
        grip = closed_fingers
    else:  # release
        u = _smoothstep((phase - 0.86) / 0.14)
        xy = target_xy
        z = 0.0
        grip = closed_fingers * (1.0 - u)

    return np.array([xy[0], xy[1], z, grip, grip, grip, grip], dtype=np.float64)

