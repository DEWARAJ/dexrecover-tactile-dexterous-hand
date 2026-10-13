from setuptools import find_packages, setup

package_name = "dexterous_hand_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dewaraj Raparthi",
    maintainer_email="dewarajraparthi@gmail.com",
    description="ROS2 nodes for the dexterous-hand learning project",
    license="MIT",
    entry_points={
        "console_scripts": [
            "simulator = dexterous_hand_ros.simulator_node:main",
            "policy = dexterous_hand_ros.policy_node:main",
        ]
    },
)
