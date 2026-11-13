# 虫洞

uint16 IDENTITY_FIELD_SET=1
uint16 POSE_FIELD_SET=2
uint16 NAV_FIELD_SET=4

# 标识
woosh_map_msgs/Identity identity
# 位姿
woosh_map_msgs/Pose pose
# 导航
woosh_map_msgs/Navigation nav
# 相同token为可互通虫洞
string token
# 虫洞所能到达的地图ID
uint32[] reach_maps
# 虫洞点所在域
uint32 wormhole_field
# 摆渡点所在域
uint32 ferry_field
# 其他域
uint32[] other_fields
# 虫洞实体
woosh_map_msgs/WormholeOneOfEntity entity
uint16 has_field 65535
