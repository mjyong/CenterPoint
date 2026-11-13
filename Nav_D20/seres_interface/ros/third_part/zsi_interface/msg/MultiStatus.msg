int8 LIGHT_STATUS_UNKNOWN = 0     # 未知
int8 LIGHT_STATUS_ON = 1          # 打开
int8 LIGHT_STATUS_OFF = 2         # 关闭

#速度等级枚举
int8 SPEED_LEVEL_UNKNOWN = 0      #未知
int8 SPEED_LEVEL_SLOW = 1         #低速
int8 SPEED_LEVEL_MEDIUM = 2       #中速
int8 SPEED_LEVEL_HIGH = 3         #高速

#机器人运动状态枚举
int8 MOTION_STATUS_UNKNOWN = 0     #未知
int8 MOTION_STATUS_STAND_UP = 1    #站立中
int8 MOTION_STATUS_LIE_DOWN = 2    #卧倒状态(趴下)
int8 MOTION_STATUS_CRAWL = 3       #匍匐状态(蹲下)
int8 MOTION_STATUS_LOCKED = 4      #锁定状态
int8 MOTION_STATUS_GENERAL = 5     #通用模式运动状态
int8 MOTION_STATUS_IN_PLACE = 6    #原地模式运动状态
int8 MOTION_STATUS_STAIR = 7       #登阶模式运动状态
int8 MOTION_STATUS_CLIMB = 8       #爬高台状态
int8 MOTION_STATUS_SLIM = 9        #瘦身状态
int8 MOTION_STATUS_GAIT = 10       #步态状态


#急停状态枚举
int8 EMERGENCY_STATUS_UNKNOWN = 0    #未知
int8 EMERGENCY_STATUS_RECOVER = 1    #急停解除
int8 EMERGENCY_STATUS_STOP = 2       #急停触发

int8 front_light_sts       #前补光灯状态 具体LIGHT_STATUS_
int8 back_light_sts        #后补光灯状态 具体LIGHT_STATUS_
bool auto_mode_light       #自动灯光模式
int8 speed_level           #当前速度等级 具体SPEED_LEVEL_
float32 speed_x            #前进/后退速度 (m/s)
float32 speed_y            #左右平移速度 (m/s) 
float32 speed_yaw          #转向角速度 (rad/s)
int8 software_emergency_sts  #软件急停状态 EMERGENCY_STATUS_
int8 hardware_emergency_sts  #硬件急停状态 EMERGENCY_STATUS_
int8 motion_status           #机器人运动状态


#底软BatteryState对应关系
#rsoc 原始电量百分比
#total_voltage  电压 (V)
#rated_current 电流 (A)
#charge_state 充放电状态   0:电源状态未知 1:正在充电 2:放电中 4:已满电
#int8[] temperature_arr 数组第一个为电池温度
