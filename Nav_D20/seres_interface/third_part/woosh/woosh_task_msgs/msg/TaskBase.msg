# 任务基础属性

# 任务ID(任务集内唯一), 范围1~99
int32 id
# 名称
string name
# 目标点编号
string mark_no
# 任务类型
woosh_task_msgs/Type type
# 动作方向
woosh_task_msgs/Direction direction
# 类型组合
int32 type_no
# 等待时间(s)
int32 wait_time
# 不可取消的任务
bool cannot_cancel
# robot.ExecTask.custom
uint8[] custom
