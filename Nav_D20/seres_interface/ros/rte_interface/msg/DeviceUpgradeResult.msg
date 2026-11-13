# DeviceUpgradeResult.msg - 设备升级执行结果消息
uint16 device_type      # 设备类型（与请求一致，见下方枚举定义）
uint8 device_id        # 设备ID（与请求一致）
uint8 result_code      # 结果代码（见下方枚举，0=成功，非0=失败）
string current_version # 升级后的实际版本号（Agent模式时填写）
bool reboot_required   # 是否需要重启设备才能生效（Agent模式时填写）

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
# result_code 结果代码枚举（uint8）
# ============================================
uint8 UPGRADE_SUCCESS=0           # 升级成功 / 资源释放成功（OTA模式）
uint8 UPGRADE_FAILED=1             # 升级失败（通用错误）
uint8 UPGRADE_TIMEOUT=2            # 升级超时
uint8 UPGRADE_CANCELLED=3          # 用户取消
uint8 UPGRADE_VERSION_MISMATCH=4  # 版本不匹配
uint8 UPGRADE_FILE_NOT_FOUND=5    # 固件文件不存在
uint8 UPGRADE_FILE_CORRUPTED=6    # 固件文件损坏
uint8 UPGRADE_DEVICE_BUSY=7       # 设备忙
uint8 UPGRADE_PERMISSION_DENIED=8 # 权限不足
uint8 UPGRADE_UNSUPPORTED_MODE=9  # 不支持的升级模式
uint8 UPGRADE_DEVICE_NOT_FOUND=10 # 设备未找到
uint8 UPGRADE_WRITE_ERROR=11      # 写入失败
uint8 UPGRADE_VERIFY_ERROR=12     # 验证失败
