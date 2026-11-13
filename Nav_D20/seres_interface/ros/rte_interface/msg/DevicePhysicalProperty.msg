uint16 device_type     # 设备类型
uint8 device_id       # 若存在多个设备，该id表示相同设备不同链路
float64 temperature   # 实时温度值，单位为摄氏度（℃）
float64 voltage       # 实时电压值，单位为伏特（V）
float64 current       # 实时电流值，单位为安培（mA）

# -----------device_type-------------
#DEV_UNKOWN = 0,              // 设备未知
#DEV_CAMERA = 0X0101,       // 相机
#DEV_GIMBAL = 0X0102,        // 云台
#DEV_LIDAR = 0X0103,          // 激光雷达
#DEV_SENSOR = 0X0104,       // 传感器(含足端、光、触觉、接近觉、六维力等)
#DEV_IMU =  0X0105,            // 惯性测量单元
#DEV_MOTOR = 0X0106,         // 驱动电机
#DEV_FAN = 0X0107,              // 风扇
#DEV_BAT = 0X0108,              // 电池
#DEV_5G = 0X0109,                // 5G
#DEV_BT_WLAN = 0X010A,             // 无线局域网
#DEV_BT = 0X010B,             // 蓝牙
#DEV_GNSS = 0X010C,         // 全球导航卫星系统
#DEV_ETHERNET = 0X010D,         // 以太网
#DEV_UWB = 0X010E,           // UWB
#DEV_EAR_MOTOR = 0X010F, // 电动耳朵
#DEV_SSD = 0x0110,                 // 固态硬盘(SD)
#DEV_SPEAKER = 0x0111,            // 扬声器
#DEV_LED = 0X0112,                // 灯控（状态灯/照明灯/氛围灯）
#DEV_CHARGE = 0x0113,           // 充电设备          
#DEV_CONTROLLER = 0X0307,      // 控制器(CCU-SOC/CCU-MCU/PMU/TMU-SOC/TMU-MCU)
#DEV_CCU_BOARD  =  0x0201,            //控制器CCU板          
#DEV_TMU_BOARD  =  0x0202,            //控制器TMU板
#DEV_PMU_BOARD  =  0x0203,            //控制器PMU板
#DEV_ALL_PERIPHERAL = 0Xffff,   // 所有外设

# -----------device_id-------------
# 驱动电机(0x0106)
# ---D30
#    右前腿: FR_01 ~ FR_03 <=> 0x01 ~ 0x03, FRW <=> 0x04 
#    左前腿: FL_01 ~ FL_03 <=> 0x11 ~ 0x13, FLW <=> 0x14
#    右后腿: RR_01 ~ RR_03 <=> 0x21 ~ 0x23, RRW <=> 0x24
#    左后腿: RL_01 ~ RL_03 <=> 0x31 ~ 0x33, RLW <=> 0x34
#    颈  部: N_Y <=> 0x41, N_R <=> 0x42, N_P <=> 0x43
# ---R11/R11-W/W11
#    右      手：0x51 ~ 0x57
#    左      手：0x61 ~ 0x67
#    右      腿：0x71 ~ 0x76
#    左      腿：0x81 ~ 0x86
#    颈      部：0x91 ~ 0x93
#    腰      部：0xA1 ~ 0xA4
#    灵巧手(右)：0xB1 ~ 0xB1
#    灵巧手(左)：0xC1 ~ 0xC1 
# 电动耳朵(0x010F)
#    耳  朵：LE <=> 0x01, RE <=> 0x02

