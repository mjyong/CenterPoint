# topic:"agv/hardware_state"
# 发布者：agv_controller
# 订阅者：
# AGV硬件状态获取

std_msgs/Header header
string break_sw_state           # 抱闸状态 ON / OFF / UNKNOWN
string laser_state              # 雷达状态 OK / ERROR / INITIALIZING / UNKNOWN
string battery_state            # 电池状态 CHARGING / NO_CHARGING / UNKNOWN
int32 battery_percentage        # 电池剩余电量，单位%
int32 battery_voltage           # 电池当前电压，单位mV
int32 battery_current           # 电池当前电流，单位mA
int32 battery_temperature       # 电池温度，单位℃
int32 battery_remain_capacity   # 电池剩余容量（mAh)
int32 battery_nominal_capacity  # 电池标称容量（mAh)
int32 battery_use_cycles        # 电池循环次数
int32 battery_remain_time       # 电池预计剩余工作时间，单位min
