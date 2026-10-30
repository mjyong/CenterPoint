from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("config", default_value="",
                              description="pipeline yaml; empty uses SERES_PERCEPTION_INSTALL_DIR"),
        DeclareLaunchArgument("model_dir", description="directory of the TensorRT engines"),
        DeclareLaunchArgument("precision", default_value="fp16"),
        DeclareLaunchArgument("threads", default_value="1"),
        DeclareLaunchArgument("local_frame", default_value="lidar_det"),
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
        }],
    )
    return LaunchDescription(args + [node])
