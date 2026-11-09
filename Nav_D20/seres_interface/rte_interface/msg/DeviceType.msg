#DeviceType.msg

uint8 device_type #设备类型
uint8 DEV_UNKOWN            = 0  # 设备未知
uint8 DEV_MCU               = 1  # MCU
uint8 DEV_SOC               = 2  # SOC
uint8 DEV_IMU               = 3  # 惯性测量单元
uint8 DEV_5G                = 4  # 5G
uint8 DEV_GNSS              = 5  # 全球导航卫星系统
uint8 DEV_BAT               = 6  # 电池
uint8 DEV_TEMP_NTC          = 7  # 温度 - 负温度系数热敏电阻
uint8 DEV_BT_WLAN           = 8  # 蓝牙/无线局域网
uint8 DEV_SSD               = 9  # 固态硬盘
uint8 DEV_EXT_DOCK          = 10 # 拓展坞
uint8 DEV_MOTOR             = 11 # 驱动电机
uint8 DEV_LIDAR             = 12 # 激光雷达
uint8 DEV_LED_STATUS_LIGHT  = 13 # 状态灯
uint8 DEV_LED_LIGHTING      = 14 # 照明灯
uint8 DEV_CAMERA            = 15 # 相机
uint8 DEV_SPEAKER           = 16 # 扬声器
uint8 DEV_FOOT_SENSOR       = 17 # 足端传感器
uint8 DEV_INTEGRATED_ANTENNA= 18 # 集成天线
uint8 DEV_SIX_AXIS_SENSOR   = 19 # 六维力传感器
uint8 DEV_TOUCH_SENSOR      = 20 # 触觉传感器
uint8 DEV_PROX_SENSOR       = 21 # 接近传感器
uint8 DEV_PMU               = 22 #电源管理单元
uint8 DEV_ALL_PERIPHERAL    = 0Xff #所有外设
