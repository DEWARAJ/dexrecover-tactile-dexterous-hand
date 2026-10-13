# ROS2 interfaces

The optional `dexterous_hand_ros` package separates simulation from policy inference.

| Interface | Type | Purpose |
|---|---|---|
| `/hand/observation` | `std_msgs/Float64MultiArray` | 29-value simulator state and task phase |
| `/hand/command` | `std_msgs/Float64MultiArray` | Seven position commands: palm XYZ and four fingers |
| `/hand/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Safety status and current target distance |
| `/hand/reset` | `std_srvs/Trigger` | Begin a randomized episode |

Build from `ros2_ws` with `colcon build --symlink-install` after installing the core Python package and sourcing a ROS2 environment. The core project is tested without ROS2; these wrappers require a ROS2 installation.
