# 导航域

# 通道id
uint32 id
# 导航模式
woosh_nav_msgs/Mode nav_mode
# 停等超时模式下的超时时间
float32 wait_time_out
# 通道最高速度, 设为0, 则不约束最高速度,
float32 max_speed
# 通道容量
uint32 capacity
# 域顶点
woosh_common_msgs/Point[] vertexs
# 域引导点
uint32[] guide_markers
