# Power management mode message
# Represents the current power state of the robot

uint8 main_mode

uint8 emergency                    #当前是否有急停
uint8 emergency_type               #急停类型，软急停\硬急停
string emergency_triggered_by      #谁触发的急停(最近一次)
uint64 emergency_timestamp         #什么时候触发的急停(最近一次)

bool   motor_power_enabled         #关节电机供电状态, true：供电正常,  false：供电异常

uint8  charge_status               #充电状态

bool   power_lock_busy             #TRUE:有节点持有电源锁,无法自动进入空闲态, FALSE:无节点持有电源锁
string power_lock_hold_by          #字符串展示哪些节点hold电源锁

##################################################################################################################
# Main modes
uint8 MAIN_MODE_SHUTDOWN     = 0   # SHUTDOWN: 在PRE_SHUTDOWN阶段等待5后自动进入,硬件/外设断电、SOC关机、电池停止供电
uint8 MAIN_MODE_SLEEP        = 1   # SLEEP: 在PRE_SLEEP阶段等待5后自动进入,硬件/外设断电
uint8 MAIN_MODE_WORK         = 2   # WORK: full functionality enabled
uint8 MAIN_MODE_PRE_SHUTDOWN = 10  # PRE_SHUTDOWN: 停止软件业务逻辑,避免进入SHUTDOWN阶段后出现软件功能异常
uint8 MAIN_MODE_PRE_SLEEP    = 11  # PRE_SLEEP: 停止软件业务逻辑,避免进入SLEEP阶段后出现软件功能异常

#emergency
uint8 EMERGENCY_EXIT  = 0
uint8 EMERGENCY_ENTER = 1

uint8 EMERGENCY_TYPE_HARD   = 0    #由硬件触发的急停,即由急停按钮被按下/硬件受到破坏触发的必须经过手动操作才能恢复的急停
uint8 EMERGENCY_TYPE_SOFT   = 1    #由软件触发的急停,即由遥控器、云端、语音触发的可以不经过手动操作即可恢复的急停

#charge_status
uint8 CHARGE_STATUS_CHARGER_NOT_CONNECTED = 0   #充电器/桩未连接
uint8 CHARGE_STATUS_MOVING_TO_CHARGER = 1       #充电桩寻找中
uint8 CHARGE_STATUS_CHARGER_CONNECTED = 2       #充电器/桩已连接
uint8 CHARGE_STATUS_CHARGING = 3                #充电执行中
