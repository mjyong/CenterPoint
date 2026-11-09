# Power management mode message
# Represents the current power state of the robot

uint8 main_mode
uint8 emergency

# Main modes
uint8 MAIN_MODE_SHUTDOWN     = 0   # SHUTDOWN: 在PRE_SHUTDOWN阶段等待5后自动进入,硬件/外设断电、SOC关机、电池停止供电
uint8 MAIN_MODE_SLEEP        = 1   # SLEEP: 在PRE_SLEEP阶段等待5后自动进入,硬件/外设断电
uint8 MAIN_MODE_WORK         = 2   # WORK: full functionality enabled
uint8 MAIN_MODE_PRE_SHUTDOWN = 10  # PRE_SHUTDOWN: 停止软件业务逻辑,避免进入SHUTDOWN阶段后出现软件功能异常
uint8 MAIN_MODE_PRE_SLEEP    = 11  # PRE_SLEEP: 停止软件业务逻辑,避免进入SLEEP阶段后出现软件功能异常

#emergency
uint8 EMERGENCY_EXIT  = 0
uint8 EMERGENCY_ENTER = 1
