uint8 device_type       # 设备类型
uint8 device_id         # 若存在多个设备，该id表示相同设备不同链路
uint8 link_status       # 链路状态,1：通信正常,0：通信断开

# ---------------------------------
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