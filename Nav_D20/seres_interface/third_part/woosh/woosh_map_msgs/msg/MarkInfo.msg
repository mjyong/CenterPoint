# 标记点信息

# 地图数据版本号, 每上传一次该版本号+1
int64 version
# SDK版本
string sdkv
# 楼层名
string floor_name
# 储位数组
woosh_map_msgs/Storage[] storages
# 虫洞数组
woosh_map_msgs/Wormhole[] wormholes
# Beacon数组
woosh_map_msgs/Beacon[] beacons
# 二维码数组
woosh_map_msgs/ARTag[] artags
# 反光板数组
woosh_map_msgs/Reflector[] reflectors
