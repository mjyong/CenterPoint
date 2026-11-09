# 机器人任务执行信息

uint8 ACTION_FIELD_SET=16

# 机器人任务ID(tset_id * 100 + task_id)
int64 robot_task_id
# 任务类型
woosh_task_msgs/Type type
# 任务状态
woosh_task_msgs/State state
# 动作信息
woosh_robot_msgs/TaskProcAction action
# 目的地
string dest
# 消息
string msg
# 最后更新时间(s)
int32 time
woosh_robot_msgs/TaskProcOneOfCommu commu
uint8 has_field 255
