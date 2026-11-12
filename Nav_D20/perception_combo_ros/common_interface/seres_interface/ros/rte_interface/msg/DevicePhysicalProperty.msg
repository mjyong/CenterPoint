uint8 device_type     # 设备类型
uint8 device_id       # 若存在多个设备，该id表示相同设备不同链路
float64 temperature   # 实时温度值，单位为摄氏度（℃）
float64 voltage       # 实时电压值，单位为伏特（V）
float64 current       # 实时电流值，单位为安培（mA）

# ----------------------------
# DEV_UNKOWN = 0, // 设备未知
# DEV_MCU = 1,    // MCU
# DEV_SOC = 2,    // SOC
# DEV_IMU = 3,    // 惯性测量单元
# DEV_5G = 4,     // 5G
# DEV_GNSS = 5,   // 全球导航卫星系统
# DEV_BAT,        // 电池
# DEV_TEMP_NTC,   // 温度 - 负温度系数热敏电阻
# DEV_BT_WLAN,    // 蓝牙/无线局域网
# DEV_SSD,        // 固态硬盘
# DEV_EXT_DOCK,   // 拓展坞
# DEV_MOTOR,      // 驱动电机
# DEV_LIDAR,      // 激光雷达
# DEV_LED_STATUS_LIGHT,  // 状态灯
# DEV_LED_LIGHTING,      // 照明灯
# DEV_CAMERA,            // 相机
# DEV_SPEAKER,           // 扬声器
# DEV_FOOT_SENSOR,       // 足端传感器
# DEV_INTEGRATED_ANTENNA,// 集成天线
# DEV_FOOT_SENSOR,       // 足端传感器
# DEV_SIX_AXIS_SENSOR,   // 六维力传感器
# DEV_TOUCH_SENSOR,      // 触觉传感器
# DEV_PROX_SENSOR,       // 接近传感器
# DEV_PMU,               //电源管理单元
# DEV_ALL_PERIPHERAL = 0Xff, //所有外设