# Beacon

uint8 POINT_FIELD_SET=4

# Beacon的Mac地址
uint32 id
# Beacon的所有分组信息
woosh_map_msgs/BeaconGroup[] group
# 位置
woosh_common_msgs/Point point
uint8 has_field 255
