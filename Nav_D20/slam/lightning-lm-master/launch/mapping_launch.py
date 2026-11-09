import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    package_path = get_package_share_directory('lightning')

    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'mapping_lightning.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')

    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    config_file = os.path.join(
        package_path, 'config', 'default_d20.yaml')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )

    lighting_loc_node = Node(
        package='lightning',
        executable='run_slam_online',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use)
    )

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('config_path', config_file))
    ld.add_action(declare_use_sim_time_cmd)

    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)

    ld.add_action(lighting_loc_node)
    ld.add_action(rviz_node)

    return ld
