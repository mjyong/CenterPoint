# 悟时任务集合

uint8 BASE_FIELD_SET=1
uint8 EXEC_FIELD_SET=2
uint8 TIME_FIELD_SET=4
uint8 CUSTOM_FIELD_SET=16
uint8 CONSTA_FIELD_SET=32

# 基础任务集属性
woosh_task_msgs/TaskSetBase base
# 任务集执行信息
woosh_task_msgs/TaskSetExec exec
# 任务集时间记录
woosh_task_msgs/TaskSetTime time
# 任务子集
woosh_task_msgs/WooshTask[] tasks
# 可变的自定义字段
woosh_task_msgs/WooshTaskSetCustom custom
# 不变的自定义字段
woosh_task_msgs/WooshTaskSetConsta consta
uint8 has_field 255
