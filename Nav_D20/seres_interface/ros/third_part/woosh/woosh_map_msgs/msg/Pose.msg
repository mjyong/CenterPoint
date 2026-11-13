# 位姿相关

uint8 DOCK_FIELD_SET=1
uint8 REAL_FIELD_SET=2

# 对接点坐标
woosh_common_msgs/Pose2D dock
# 实际坐标
woosh_common_msgs/Pose2D real
# 调整
woosh_map_msgs/PoseOneOfCategory category
uint8 has_field 255
