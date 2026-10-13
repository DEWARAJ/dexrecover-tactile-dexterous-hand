import numpy as np

from dexterous_hand_learning.safety import SafetyMonitor


def test_accepts_valid_command():
    monitor = SafetyMonitor(np.zeros(2), np.ones(2), 0.1)
    result = monitor.validate(np.array([0.25, 0.75]), 0.02)
    assert result.safe
    assert result.reason == "ok"


def test_rejects_non_finite_and_stale_commands():
    monitor = SafetyMonitor(np.zeros(2), np.ones(2), 0.1)
    assert monitor.validate(np.array([np.nan, 0.2])).reason == "non_finite_command"
    assert monitor.validate(np.array([0.2, 0.2]), 0.2).reason == "stale_command"


def test_clips_limit_violation():
    monitor = SafetyMonitor(np.zeros(2), np.ones(2), 0.1)
    result = monitor.validate(np.array([-0.5, 1.5]))
    assert not result.safe
    np.testing.assert_allclose(result.command, [0.0, 1.0])

