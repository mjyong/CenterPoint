float64 time # 目标时间戳
uint32  id # 目标id
int8 valid  # 目标是否有效 0 无效 1有效
float32 x  # 相对坐标
float32 y  # 相对坐标
float32 heading  # 目标运动航向（机器人坐标系，弧度，atan2(vy,vx)，静止时保持上一航向）
float32 xabs  # 绝对坐标
float32 yabs  # 绝对坐标
int8 searching_mode # 是否开启目标搜索模式 0 关闭 1开启
float32 vx # 目标速度x分量（机器人坐标系，m/s，由绝对位置差分自估计）
float32 vy # 目标速度y分量（机器人坐标系，m/s，由绝对位置差分自估计）