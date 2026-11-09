# 导航模式设定

# 导航到达类型
woosh_nav_msgs/ArrType type
# 导航模式
woosh_nav_msgs/Mode mode
# nav_mode为kTimeout时有效, 该参数指定超时时间(秒)
uint32 wait_timeout
# 导航的最大速度, 为0时取默认速度
float32 max_speed
# 是否允许通行
bool permitted_passage
# 通道车量
int32 capacity
