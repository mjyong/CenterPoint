# 呼叫事件

uint8 CALLER_FIELD_SET=1
uint8 GUARD_FIELD_SET=2

# 呼叫
woosh_task_msgs/CallInfoCaller caller
# 呼叫守护
woosh_task_msgs/CallInfoCallGuard guard
# 任务ID, 任务完成释放呼叫器
uint32 release
uint8 has_field 255
