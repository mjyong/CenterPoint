# 场景数据

uint8 TASK_INFO_FIELD_SET=4

# 场景名
string name
# 地图信息数据
woosh_map_msgs/SceneDataMap[] maps
# task_info.data
woosh_task_msgs/TaskInfo task_info
# 动作组合json
string action_group
uint8 has_field 255
