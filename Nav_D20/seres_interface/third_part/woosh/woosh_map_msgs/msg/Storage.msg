# 储位

uint16 IDENTITY_FIELD_SET=1
uint16 POSE_FIELD_SET=2
uint16 NAV_FIELD_SET=4
uint16 DOCK_FIELD_SET=8

# 标识
woosh_map_msgs/Identity identity
# 位姿
woosh_map_msgs/Pose pose
# 导航
woosh_map_msgs/Navigation nav
# 对接
woosh_map_msgs/Dock dock
# 自定义字段
uint8[] custom
# 类别
woosh_map_msgs/StorageOneOfCategory category
uint16 has_field 65535
