# 插入任务

uint8 TASK_FIELD_SET=2

# 任务集ID
int32 tset_id
# 要插入的任务, 任务ID需要确保任务集合内唯一
woosh_task_msgs/WooshTask task
uint8 has_field 255
