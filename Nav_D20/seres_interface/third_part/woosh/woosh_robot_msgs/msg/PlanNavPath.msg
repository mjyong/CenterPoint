# 规划导航路径请求

uint8 START_FIELD_SET=2
uint8 END_FIELD_SET=4

# 起始点
woosh_common_msgs/Pose2D start
# 目标点
woosh_common_msgs/Pose2D end
# 允许绕行范围, 单位为米
float32 tolerance
woosh_robot_msgs/PlanNavPathOneOfCommu commu
uint8 has_field 255
