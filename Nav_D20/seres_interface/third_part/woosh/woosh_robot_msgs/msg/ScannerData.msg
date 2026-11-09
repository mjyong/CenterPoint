# 雷达点云信息

uint16 POSE_FIELD_SET=512
uint16 OFFSET_FIELD_SET=1024

# scan的开始角度 [弧度]
float32 angle_min
# scan的结束角度 [弧度]
float32 angle_max
# 测量的角度间的距离 [弧度]
float32 angle_increment
# 测量间的时间 [秒]
float32 time_increment
# 扫描间的时间 [秒]
float32 scan_time
# 最小的测量距离 [米]
float32 range_min
# 最大的测量距离 [米]
float32 range_max
# 测量的距离数据 [米] (注意: 值 < range_min 或 > range_max 应当被丢弃)
float32[] ranges
# 位姿
woosh_common_msgs/Pose2D pose
# 偏移量
woosh_common_msgs/Vector3 offset
woosh_robot_msgs/ScannerDataOneOfCommu commu
uint16 has_field 65535
