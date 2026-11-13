# DeviceUpgradeRequest.msg 设备升级列表消息
uint16 device_type     # 设备类型（枚举定义）
uint8 device_id       # 设备唯一标识符
uint8 upgrade_mode    # 升级模式（0=OTA自升级, 1=底软辅助升级）
string firmware_path  # 固件文件绝对路径（含文件名）
string target_version # 目标版本号（可选，为空则跳过版本检查）
bool force_upgrade    # 是否强制升级（默认false）

# ============================================
# device_type 设备类型枚举（uint16）
# ============================================
uint16  DEV_CAMERA = 0X0101         # 相机
uint16  DEV_GIMBAL = 0X0102         # 云台
uint16  DEV_LIDAR = 0X0103          # 激光雷达
uint16  DEV_SENSOR = 0X0104         # 传感器(含足端、光、触觉、接近觉、六维力等)
uint16  DEV_IMU =  0X0105           # 惯性测量单元
uint16  DEV_MOTOR = 0X0106          # 驱动电机
uint16  DEV_FAN = 0X0107            # 风扇
uint16  DEV_BAT = 0X0108            # 电池
uint16  DEV_5G = 0X0109             # 5G
uint16  DEV_BT_WLAN = 0X010A        # 无线局域网
uint16  DEV_BT = 0X010B             # 蓝牙
uint16  DEV_GNSS = 0X010C           # 全球导航卫星系统
uint16  DEV_ETHERNET = 0X010D       # 以太网
uint16  DEV_UWB = 0X010E            # UWB
uint16  DEV_EAR_MOTOR = 0X010F      # 电动耳朵
uint16  DEV_SSD = 0x0110            # 固态硬盘(SD)
uint16  DEV_SPEAKER = 0x0111        # 扬声器
uint16  DEV_LED = 0X0112            # 灯控（状态灯/照明灯/氛围灯）
uint16  DEV_CHARGE = 0x0113         # 充电设备
uint16  DEV_CONTROLLER = 0x0307     #控制器(CCU-SOC/CCU-MCU/PMU/TAM-SOC/TMU-MCU)
uint16  DEV_CCU_BOARD = 0x0201  # CCU板电压采集
uint16  DEV_TMU_BOARD = 0X0202  # TMU板电压采集
uint16  DEV_PMU_BOARD = 0X0203  # PMU板电压采集

# ============================================
# upgrade_mode 升级模式枚举（uint8）
# ============================================
uint8 UPGRADE_MODE_OTA=0         # OTA 自升级模式
uint8 UPGRADE_MODE_AGENT=1       # 底软辅助升级模式 (AGENT)
