from pathlib import Path

import numpy as np

from dexterous_hand_learning.articulated import AllegroHandEnvironment
from dexterous_hand_learning.tactile import TactileRecoveryController, TactileState


ROOT = Path(__file__).resolve().parents[1]


def test_slip_event_increases_bounded_grip():
    env = AllegroHandEnvironment(ROOT / "third_party" / "wonik_allegro" / "scene_right.xml")
    controller = TactileRecoveryController(env)
    command = controller.update(TactileState(0, 0.0, 0.1, True))
    assert np.all(command >= env.low)
    assert np.all(command <= env.high)
    assert command[1] > 0.75
