# 执行任务请求

uint16 PLAN_PATH_FIELD_SET=32
uint16 POSE_FIELD_SET=64
uint16 NAV_MODE_FIELD_SET=512

# 任务ID
int64 task_id
# 任务类型
woosh_task_msgs/Type type
# 动作方向
woosh_task_msgs/Direction direction
# 类型组合
uint32 task_type_no
# 目标点编号(三选一)
string mark_no
# 导航路径集合(三选一)
woosh_robot_msgs/PlanPath plan_path
# 位姿(三选一)
woosh_common_msgs/Pose2D pose
# 目标点地图ID, 多地图发位姿时用
uint32 map_id
# 等待时间(s), -1则无限等待, action.Order.kWaitBreak结束等待
int32 wait_time
# 导航模式设置
woosh_nav_msgs/ModeSetting nav_mode
# 自定义字段, 因项目而异
uint8[] custom
uint16 has_field 65535
