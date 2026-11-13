# 任务集

uint8 BASE_FIELD_SET=1
uint8 EXEC_FIELD_SET=2
uint8 TIME_FIELD_SET=4

# 任务集基础属性
woosh_task_msgs/TaskSetBase base
# 任务集执行信息
woosh_task_msgs/TaskSetExec exec
# 任务集时间记录
woosh_task_msgs/TaskSetTime time
# 所有任务
woosh_task_msgs/Task[] tasks
uint8 has_field 255
