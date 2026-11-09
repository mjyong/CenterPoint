uint16 device_type       # 设备类型
uint8 device_id         # 若存在多个设备，该id表示相同设备不同链路
uint8 link_status       # 链路状态,1：通信正常,0：通信断开

# --------------d30------------
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
#DEV_ALL_PERIPHERAL = 0Xffff,   // 所有外设

# --------------other------------
# DEV_UNKOWN = 0, // 设备未知
# DEV_MCU = 1,    // MCU
# DEV_SOC = 2,    // SOC
# DEV_IMU = 3,    // 惯性测量单元
# DEV_5G = 4,     // 5G
# DEV_GNSS = 5,   // 全球导航卫星系统
# DEV_BAT = 6,        // 电池
# DEV_TEMP_NTC = 7,   // 温度 - 负温度系数热敏电阻
# DEV_BT_WLAN = 8,    // 蓝牙/无线局域网
# DEV_SSD = 9,        // 固态硬盘
# DEV_EXT_DOCK = 10,   // 拓展坞
# DEV_MOTOR = 11,      // 驱动电机
# DEV_LIDAR = 12,      // 激光雷达
# DEV_LED_STATUS_LIGHT = 13,  // 状态灯
# DEV_LED_LIGHTING = 14,      // 照明灯
# DEV_FAN = 15,            // 风扇
# DEV_CAMERA = 16 ,            // 相机
# DEV_SPEAKER = 17,           // 扬声器
# DEV_FOOT_SENSOR = 18,       // 足端传感器
# DEV_INTEGRATED_ANTENNA = 19,// 集成天线
# DEV_FOOT_SENSOR = 20,       // 足端传感器
# DEV_SIX_AXIS_SENSOR = 21,   // 六维力传感器
# DEV_TOUCH_SENSOR = 22,      // 触觉传感器
# DEV_PROX_SENSOR = 23,       // 接近传感器
# DEV_PMU = 24, //电源管理单元
# DEV_ALL_PERIPHERAL = 0Xff, //所有外设