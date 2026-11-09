# 循环任务

uint8 TASK_SET_FIELD_SET=2

# 重复次数
int32 times
# 任务集合
woosh_task_msgs/WooshTaskSet task_set
# RF433.Addr
uint32 addr
# RF433.Key
uint32 key
# 指令
woosh_task_msgs/RepeatTaskOrder order
uint8 has_field 255
