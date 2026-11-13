std_msgs/Header header           # ROS2 标准消息头
uint16 dev_type                  # 外设（控制器）类型
uint8 dev_index                  # 设备序号（当存在多个相同设备单独供电时）
uint8 duty_set                   # 设置转速（占空比）
uint8 duty_real                  # 实际转速（占空比）
