# 悟时任务

uint8 BASE_FIELD_SET=1
uint8 EXEC_FIELD_SET=2
uint8 CUSTOM_FIELD_SET=4
uint8 CONSTA_FIELD_SET=8

# 基础任务属性
woosh_task_msgs/TaskBase base
# 任务执行信息
woosh_task_msgs/TaskExec exec
# 可变的自定义字段
woosh_task_msgs/WooshTaskCustom custom
# 不变的自定义字段
woosh_task_msgs/WooshTaskConsta consta
uint8 has_field 255
