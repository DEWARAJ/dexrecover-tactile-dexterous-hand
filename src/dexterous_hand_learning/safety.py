from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyResult:
    command: np.ndarray
    safe: bool
    reason: str


class SafetyMonitor:
    """Validates actuator commands before they reach the simulator or robot."""

    def __init__(self, low: np.ndarray, high: np.ndarray, max_action_age_s: float = 0.1):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.max_action_age_s = float(max_action_age_s)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high limits must have the same shape")

    def validate(self, command: np.ndarray, action_age_s: float = 0.0) -> SafetyResult:
        command = np.asarray(command, dtype=np.float64)
        neutral = (self.low + self.high) / 2.0
        if command.shape != self.low.shape:
            return SafetyResult(neutral, False, "shape_mismatch")
        if not np.all(np.isfinite(command)):
            return SafetyResult(neutral, False, "non_finite_command")
        if action_age_s > self.max_action_age_s:
            return SafetyResult(neutral, False, "stale_command")
        clipped = np.clip(command, self.low, self.high)
        if not np.allclose(clipped, command):
            return SafetyResult(clipped, False, "joint_limit_clip")
        return SafetyResult(command, True, "ok")

