# 自动门域

uint8 NAV_FILED_FIELD_SET=1
uint8 IN_POSE_FIELD_SET=2
uint8 OUT_POSE_FIELD_SET=4

# 导航域信息
woosh_map_msgs/NavField nav_filed
# 入口
woosh_map_msgs/GuidePose in_pose
# 出口
woosh_map_msgs/GuidePose out_pose
# 路径网节点
woosh_map_msgs/GuidePoint[] guide_points
# 半径
float32 radius
# 偏移
float32 offset
# 自定义字段(IP等)
uint8[] custom
uint8 has_field 255
