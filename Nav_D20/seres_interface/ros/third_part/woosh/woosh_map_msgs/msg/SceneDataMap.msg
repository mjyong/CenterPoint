# 地图数据

uint16 ORIGIN_FIELD_SET=8
uint16 END_FIELD_SET=16
uint16 MARK_INFO_FIELD_SET=256
uint16 FIELD_INFO_FIELD_SET=512
uint16 PATH_INFO_FIELD_SET=1024

# 地图ID
uint32 id
# 地图名称
string name
# 地图分辨率
float32 resolution
# 原点坐标
woosh_common_msgs/Pose2D origin
# 构图结束点
woosh_common_msgs/Pose2D end
# 地图PNG
uint8[] map_png
# 导航禁区PNG
uint8[] keepout_png
# 交管禁区PNG
uint8[] traffic_keepout_png
# mark_info.data
woosh_map_msgs/MarkInfo mark_info
# field_info.data
woosh_map_msgs/FieldInfo field_info
# path_info.data
woosh_map_msgs/PathInfo path_info
# 文件最后一次被修改的时间
string time_write
uint16 has_field 65535
