from pathlib import Path

import numpy as np

from dexterous_hand_learning.articulated import AllegroHandEnvironment, smooth_grasp_command


ROOT = Path(__file__).resolve().parents[1]


def test_official_allegro_model_loads_and_steps():
    env = AllegroHandEnvironment(ROOT / "third_party" / "wonik_allegro" / "scene_right.xml")
    observation = env.reset(seed=3)
    assert env.model.nu == 16
    assert observation.shape == (35,)
    observation = env.step(smooth_grasp_command(1, 10))
    assert np.all(np.isfinite(observation))
