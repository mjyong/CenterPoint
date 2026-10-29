"""The dog perception node (deploy/ros2/dogperc_ros).

    ros2 launch dogperc_ros dogperc.launch.py config:=$HOME/dogperc/kygi.yaml \
        model_dir:=$HOME/dogperc/models/nusc_pp_dyn

config: tools/deploy/export_cpp_config.py --robot kygi (topics, localization and preprocessing of the robot);
model_dir: the exported model with its engines built on this Orin (build_engines.py).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("config", description="YAML of tools/deploy/export_cpp_config.py --robot"),
        DeclareLaunchArgument("model_dir", description="exported model directory with its engines"),
        DeclareLaunchArgument("precision", default_value="fp16"),
        DeclareLaunchArgument("threads", default_value="4",
                              description="threads of the per-point preprocessing (4 on the Orin, T28B)"),
        DeclareLaunchArgument("local_frame", default_value="dog_det", description="frame_id of ~/objects"),
    ]
    node = Node(package="dogperc_ros", executable="dogperc_node", name="dog_perception", output="screen",
                parameters=[{
                    "config": LaunchConfiguration("config"),
                    "model_dir": LaunchConfiguration("model_dir"),
                    "precision": LaunchConfiguration("precision"),
                    "gpu": True,
                    "threads": ParameterValue(LaunchConfiguration("threads"), value_type=int),
                    "local_frame": LaunchConfiguration("local_frame"),
                }])
    return LaunchDescription(args + [node])
