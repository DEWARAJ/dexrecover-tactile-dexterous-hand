from pathlib import Path

import numpy as np

from dexterous_hand_learning.environment import HandEnvironment


ROOT = Path(__file__).resolve().parents[1]


def test_environment_reset_and_step_are_finite():
    env = HandEnvironment(ROOT / "assets" / "four_finger_hand.xml", episode_steps=5)
    observation = env.reset(seed=1)
    assert observation.shape == (env.observation_size,)
    assert env.observation_size == 29
    assert np.all(np.isfinite(observation))
    result = env.step(np.zeros(7), phase=0.0)
    assert np.all(np.isfinite(result.observation))
    assert result.info["safety_reason"] == "ok"


def test_randomized_resets_are_deterministic_by_seed():
    env = HandEnvironment(ROOT / "assets" / "four_finger_hand.xml")
    env.reset(seed=44)
    first = np.r_[env.cube_position()[:2], env.target_xy]
    env.reset(seed=44)
    second = np.r_[env.cube_position()[:2], env.target_xy]
    np.testing.assert_allclose(first, second)
