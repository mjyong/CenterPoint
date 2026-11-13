# 斜坡域

uint8 NAV_FILED_FIELD_SET=1
uint8 ARROW_POSE_FIELD_SET=4

# 导航域信息
woosh_map_msgs/NavField nav_filed
# 爬坡域角度
float32 slope_angle
# 爬坡域箭头位姿
woosh_common_msgs/Pose2D arrow_pose
uint8 has_field 255
