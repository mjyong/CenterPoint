std_msgs/Header header

# 状态数组，按顺序包含：
# 0: 当前转态：摔倒、正常、。。等
# 1: motor_signal_interruption - 当前是否电机信号中断
uint8[] mc_status_array