# 初始化机器人请求

uint8 POSE_FIELD_SET=4

# 是否记录点复位, 三选一
bool is_record
# 是否二维码复位, 三选一
bool is_qrcode
# 设置机器人为新的坐标, 三选一
woosh_common_msgs/Pose2D pose
uint8 has_field 255
