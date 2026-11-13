# 单个状态统计信息

uint8 INFO_FIELD_SET=4
uint8 BUCKET_FIELD_SET=8

# 唯一标识
uint32 id
# 状态信息
woosh_robot_msgs/StatusCode info
# 触发时间段
woosh_common_msgs/TimeBucket bucket
# 恢复耗时
uint32 cons
# 结束标识
bool end
woosh_robot_msgs/StatusOneOfCommu commu
uint8 has_field 255
