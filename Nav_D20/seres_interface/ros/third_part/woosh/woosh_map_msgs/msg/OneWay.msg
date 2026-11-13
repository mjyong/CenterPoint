# 单向道

uint8 NAV_FILED_FIELD_SET=1

# 导航域信息
woosh_map_msgs/NavField nav_filed
# 单向道的方向
woosh_map_msgs/OneWayDirection direction
# 路径网节点, 单向道中引导点的存储顺序决定其方向
woosh_map_msgs/GuidePoint[] guide_points
uint8 has_field 255
