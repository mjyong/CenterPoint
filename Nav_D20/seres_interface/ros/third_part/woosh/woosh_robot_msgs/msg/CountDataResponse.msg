# 统计运行应答

uint8 OPERATIONS_FIELD_SET=1
uint8 TASKS_FIELD_SET=2
uint8 ABNORMALS_FIELD_SET=4

# 统计运行信息
woosh_robot_msgs/Operations operations
# 统计任务信息
woosh_robot_msgs/Tasks tasks
# 统计状态信息
woosh_robot_msgs/Abnormals abnormals
# 运行总数量
int32 operation_qty
# 任务总数量
int32 task_qty
# 状态总数量
int32 abnormal_qty
uint8 has_field 255
