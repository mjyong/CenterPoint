# 相机点云(稀疏压缩后的)

uint8 POSE_FIELD_SET=4
uint8 OFFSET_FIELD_SET=8

# 点云
woosh_common_msgs/Point[] points
# 位姿
woosh_common_msgs/Pose2D pose
# 偏移量
woosh_common_msgs/Vector3 offset
woosh_robot_msgs/CameraDataOneOfCommu commu
uint8 has_field 255
