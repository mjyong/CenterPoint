# 单个运作统计信息

uint8 MILEAGE_FIELD_SET=8

# 唯一标识
uint32 id
# 日期
uint32 date
# 行驶里程
woosh_robot_msgs/OperationMileage mileage
# 开机时间段
woosh_common_msgs/TimeBucket[] buckets
# 累计里程
uint32 distance
# 累计运行时长
uint32 uptime
# 结束标识
bool end
woosh_robot_msgs/OperationOneOfCommu commu
uint8 has_field 255
