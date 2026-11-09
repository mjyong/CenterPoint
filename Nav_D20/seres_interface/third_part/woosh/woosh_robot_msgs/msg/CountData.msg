# 统计运行请求

uint16 BUCKET_FIELD_SET=4

# 请求类型
woosh_robot_msgs/CountDataType req_type
# 统计类型(获取)
woosh_robot_msgs/CountDataCountType count_type
# 请求时间段(获取)
woosh_common_msgs/TimeBucket bucket
# 唯一标识(同步)
uint32[] operation_id
# 唯一标识(同步)
uint32[] task_id
# 唯一标识(同步)
uint32[] status_id
# 运行分页, 未指定则返回第0页100条
int32 operation_page
# 任务分页, 未指定则返回第0页100条
int32 task_page
# 状态分页, 未指定则返回第0页100条
int32 status_page
uint16 has_field 65535
