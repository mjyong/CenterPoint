# 改变导航模式请求

uint8 NAV_MODE_FIELD_SET=1
uint8 IN_POINT_FIELD_SET=2
uint8 OUT_POINT_FIELD_SET=4

# 导航模式设置
woosh_nav_msgs/ModeSetting nav_mode
# 域入口点
woosh_common_msgs/Pose2D in_point
# 域出口点
woosh_common_msgs/Pose2D out_point
uint8 has_field 255
