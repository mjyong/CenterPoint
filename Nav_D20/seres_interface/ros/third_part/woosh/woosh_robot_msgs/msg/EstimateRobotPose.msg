# 估算机器人位姿请求

uint8 MAP_RANGE_MIN_FIELD_SET=2
uint8 MAP_RANGE_MAX_FIELD_SET=4
uint8 POSE_FIELD_SET=8

# 方法
woosh_robot_msgs/EstimateRobotPoseMethod method
# 矩形范围的最小角（x_min, y_min）
woosh_common_msgs/Pose2D map_range_min
# 矩形范围的最大角（x_max, y_max）
woosh_common_msgs/Pose2D map_range_max
# 估算坐标
woosh_common_msgs/Pose2D pose
# 置信度, 范围[0, 100]
uint32 confidence
uint8 has_field 255
