# 机器人位姿速度

uint8 TWIST_FIELD_SET=2
uint8 POSE_FIELD_SET=4

# 速度
woosh_common_msgs/Twist twist
# 位姿
woosh_common_msgs/Pose2D pose
# 地图ID
uint32 map_id
# 累计里程, 单位m
uint32 mileage
woosh_robot_msgs/PoseSpeedOneOfCommu commu
uint8 has_field 255
