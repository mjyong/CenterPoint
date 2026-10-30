from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("config", default_value="",
                              description="pipeline yaml; empty: the installed config/lidar_3d_detection.yaml"),
        DeclareLaunchArgument("model_dir", default_value="",
                              description="directory of the TensorRT engines; empty: node.model_dir of the config"),
        DeclareLaunchArgument("precision", default_value="fp16"),
        DeclareLaunchArgument("threads", default_value="4",
                              description="threads of the per-point preprocessing (4 on the Orin)"),
        DeclareLaunchArgument("local_frame", default_value="lidar_det"),
        DeclareLaunchArgument("publish_tf", default_value="false",
                              description="TF for viewers on a replayed bag (not on the vehicle: its TF tree has the lidar)"),
    ]
    node = Node(
        package="lidar_3d_detection",
        executable="lidar_3d_detection",
        name="lidar_3d_detection",
        output="screen",
        parameters=[{
            "config": LaunchConfiguration("config"),
            "model_dir": LaunchConfiguration("model_dir"),
            "precision": LaunchConfiguration("precision"),
            "gpu": True,
            "threads": ParameterValue(LaunchConfiguration("threads"), value_type=int),
            "local_frame": LaunchConfiguration("local_frame"),
            "publish_tf": ParameterValue(LaunchConfiguration("publish_tf"), value_type=bool),
        }],
    )
    return LaunchDescription(args + [node])
