from __future__ import annotations

import numpy as np


class TeleoperationController:
    """Incremental Cartesian/gripper command generator for a keyboard or gamepad."""

    KEY_DELTAS = {
        "w": (0, 1.0),
        "s": (0, -1.0),
        "a": (1, 1.0),
        "d": (1, -1.0),
        "r": (2, 1.0),
        "f": (2, -1.0),
    }

    def __init__(self, low: np.ndarray, high: np.ndarray, initial: np.ndarray):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.command = np.clip(np.asarray(initial, dtype=np.float64), self.low, self.high)

    def update(self, key: str, translation_step: float = 0.005, grip_step: float = 0.003) -> np.ndarray:
        key = key.lower()
        if key in self.KEY_DELTAS:
            axis, direction = self.KEY_DELTAS[key]
            self.command[axis] += direction * translation_step
        elif key == "g":
            self.command[3:] += grip_step
        elif key == "o":
            self.command[3:] -= grip_step
        elif key == "n":
            self.command = (self.low + self.high) / 2.0
        self.command = np.clip(self.command, self.low, self.high)
        return self.command.copy()
