# 跟随请求

uint8 POSE_FIELD_SET=4

# TRUE:开启自动跟随, FALSE:关闭自动跟随
bool enable
# 跟随模式
woosh_robot_msgs/FollowMode mode
# 位姿(base_link坐标系)
woosh_common_msgs/Pose2D pose
uint8 has_field 255
