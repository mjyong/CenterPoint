# 自动对接

uint8 TARGET_POSE_FIELD_SET=8

# 特征板类型
woosh_ros_msgs/AutoDockFeatureBoardType board_type
# 识别tag
string tag
# 对接方向
woosh_ros_msgs/AutoDockDockDirection dock_direction
# 目标点
woosh_common_msgs/Pose2D target_pose
woosh_ros_msgs/ControlAction action
uint8 has_field 255
