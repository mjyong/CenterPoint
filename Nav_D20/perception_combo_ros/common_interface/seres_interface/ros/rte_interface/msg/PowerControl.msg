# 电源管理工作模式
uint8 power_mode             # 对应 WorkMode_e

# 控制消息数组
PowerStatus[] ctrl_msgs

# --------------------
# PowerMode 常量定义
uint8 POWER_MODE_ACTIVE         = 0    # 激活模式
uint8 POWER_MODE_SLEEP          = 1    # 休眠模式
uint8 POWER_MODE_SHUTDOWN       = 2    # 停止模式
uint8 POWER_MODE_EMERGENCY_STOP = 3    # 急停模式

# --------------------
# PowerStatus字段说明：表示多个控制消息的状态
