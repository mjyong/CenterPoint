# 地图数据

uint8 STORAGES_FIELD_SET=16

# 地图ID
uint32 id
# 地图名
string name
# 楼层名
string floor
# 地图版本
int64 version
# 储位集
woosh_map_msgs/Storages storages
# 虫洞集
woosh_map_msgs/Wormhole[] wormholes
# 二维码集
woosh_map_msgs/ARTag[] artags
# 反光板集
woosh_map_msgs/Reflector[] reflectors
uint8 has_field 255
