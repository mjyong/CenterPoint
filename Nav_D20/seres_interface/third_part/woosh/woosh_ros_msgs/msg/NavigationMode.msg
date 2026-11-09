# 导航模式

# 导航ID(弃用)
uint32 id
woosh_ros_msgs/NavigationModeMode mode
# 导航模式为等待避让时, 时间必须大于0
float32 wait_time
# 最大导航速度
float32 max_speed
# 如果允许则按照新的路径行走, 否则不允许沿变更路径行走
bool permitted_passage
# 通行车量
int32 capacity
