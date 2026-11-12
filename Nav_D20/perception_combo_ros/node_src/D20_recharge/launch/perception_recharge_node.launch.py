import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('perception_recharge'),
        'config',
        'perception_recharge_node.yaml')

    return LaunchDescription([
        Node(
            package='perception_recharge',
            executable='perception_recharge_node',
            name='perception_recharge_node',
            parameters=[config_file],
            output='screen',
        ),
    ])
