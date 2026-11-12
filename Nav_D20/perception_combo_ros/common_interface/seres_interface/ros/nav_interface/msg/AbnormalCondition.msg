std_msgs/Header header

# 状态数组，按顺序包含：
# 0: low_battery - 当前是否为低电量（运控使用）
# 1: motor_overheating - 当前是否为电机过温（运控使用）
# 2: fall_over - 当前是否摔倒（任务调度使用）
# 3: motor_signal_interruption - 当前是否电机信号中断（任务调度使用）

bool[] status_array