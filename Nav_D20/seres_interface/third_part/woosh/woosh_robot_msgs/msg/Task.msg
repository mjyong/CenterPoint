# 单个任务统计信息

uint8 BUCKET_FIELD_SET=32

# 唯一标识
uint32 id
# 任务id
int64 task_id
# 目的地
string dest
# 任务状态
woosh_robot_msgs/TaskState state
# 任务时间段
woosh_common_msgs/TimeBucket bucket
# 任务耗时
uint32 cons
# 结束标识
bool end
woosh_robot_msgs/TaskOneOfCommu commu
uint8 has_field 255
