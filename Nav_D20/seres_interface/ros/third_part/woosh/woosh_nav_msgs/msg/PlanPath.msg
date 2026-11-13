# 规划的路径

uint8 PATH_FIELD_SET=1
uint8 TARGET_FIELD_SET=16

# 导航路径, 不能为空, 只有一个值则表示由单机自主规划路径
woosh_nav_msgs/Path path
# 路径所在地图ID
uint32 map_id
# 路径去往虫洞ID, 虫洞ID为0则表示该路径不经过虫洞
uint32 wormhole_id
# 虫洞到达的地图ID
uint32 dest_map_id
# 分段目标点
woosh_common_msgs/Pose2D target
# 路径优化
woosh_nav_msgs/PlanPathOptimal optimal
uint8 has_field 255
