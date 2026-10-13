from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from dexterous_hand_learning.policy import NormalizedPolicy


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("dexterous_hand_policy")
        root = Path(__file__).resolve().parents[4]
        self.policy = NormalizedPolicy.load(root / "artifacts" / "reference" / "behavior_cloning.pt")
        self.publisher = self.create_publisher(Float64MultiArray, "/hand/command", 10)
        self.create_subscription(Float64MultiArray, "/hand/observation", self.on_observation, 10)

    def on_observation(self, message: Float64MultiArray) -> None:
        command = self.policy(np.asarray(message.data, dtype=np.float32))
        output = Float64MultiArray()
        output.data = command.astype(float).tolist()
        self.publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
