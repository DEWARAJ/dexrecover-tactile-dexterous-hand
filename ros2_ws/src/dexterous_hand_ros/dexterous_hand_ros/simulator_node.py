from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from dexterous_hand_learning.environment import HandEnvironment


class SimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("dexterous_hand_simulator")
        root = Path(__file__).resolve().parents[4]
        self.env = HandEnvironment(root / "assets" / "four_finger_hand.xml")
        self.observation = self.env.reset(seed=7, randomize=True)
        self.command = self.env.data.ctrl.copy()
        self.phase = 0.0
        self.observation_pub = self.create_publisher(Float64MultiArray, "/hand/observation", 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/hand/diagnostics", 10)
        self.create_subscription(Float64MultiArray, "/hand/command", self.on_command, 10)
        self.create_service(Trigger, "/hand/reset", self.on_reset)
        self.create_timer(self.env.control_dt, self.tick)

    def on_command(self, message: Float64MultiArray) -> None:
        self.command = np.asarray(message.data, dtype=np.float64)

    def on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.observation = self.env.reset(randomize=True)
        self.phase = 0.0
        response.success = True
        response.message = "randomized episode reset"
        return response

    def tick(self) -> None:
        result = self.env.step(self.command, self.phase)
        self.observation = result.observation
        self.phase = min(1.0, self.phase + 1.0 / max(1, self.env.episode_steps - 1))
        observation = Float64MultiArray()
        observation.data = self.observation.astype(float).tolist()
        self.observation_pub.publish(observation)
        diagnostics = DiagnosticArray()
        status = DiagnosticStatus()
        status.name = "dexterous_hand_command_safety"
        status.level = DiagnosticStatus.OK if result.info["safety_reason"] == "ok" else DiagnosticStatus.WARN
        status.message = str(result.info["safety_reason"])
        status.values = [KeyValue(key="distance_m", value=str(result.info["distance_m"]))]
        diagnostics.status = [status]
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        self.diagnostics_pub.publish(diagnostics)


def main() -> None:
    rclpy.init()
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
