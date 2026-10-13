import numpy as np

from dexterous_hand_learning.teleoperation import TeleoperationController


def test_keyboard_mapping_and_limits():
    low = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    high = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    controller = TeleoperationController(low, high, np.zeros(7))
    assert controller.update("w")[0] > 0
    assert controller.update("a")[1] > 0
    assert np.all(controller.update("g")[3:] > 0)
    for _ in range(100):
        command = controller.update("g")
    assert np.allclose(command[3:], high[3:])
