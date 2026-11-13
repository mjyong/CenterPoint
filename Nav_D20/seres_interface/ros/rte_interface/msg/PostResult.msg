std_msgs/Header header 
# ==========================
# 1. 电源相关
# ==========================
bool battery_voltage_ok          # 检查项：电池供电检测；true = 电压在标准范围内，false = 超差或未检测到
bool pmu_ok                      # 检查项：PMU 供电检测；true = PMU 内部 DCDC/高边驱动均正常，false = 异常
bool ccu_power_ok                # 检查项：CCU 供电检测；true = CCU 内部 DCDC/高边驱动均正常，false = 异常


# ==========================
# 2. CCU 核心状态
# ==========================
bool mcu_running                 # 检查项：MCU 运行状态检测；true = MCU 已正常启动并持续运行，false = 未启动/死机
bool soc_running                 # 检查项：SOC 运行状态监测；true = SOC 已正常启动并持续运行，false = 未启动/死机
bool memory_ok                   # 检查项：内存检测；true = 内存读写全部通过，false = 存在 bit-error/无法访问
bool sw_system_ok                # 检查项：软件系统状态检查；true = CCU 软件系统正常拉起且无异常重启，false = 启动失败/异常重启


# ==========================
# 3. 集成天线通信
# ==========================
bool antenna_5g_ok               # 检查项：集成天线 5G 功能检查；true = 已注册 5G 网络，false = 无信号/注册失败
bool antenna_gnss_ok             # 检查项：集成天线 GNSS 功能检查；true = 已获取有效 GNSS 定位，false = 无定位/数据异常
bool antenna_bt_ok               # 检查项：集成天线蓝牙功能检查；true = 蓝牙模块可扫描/被连接，false = 模块未响应
bool antenna_wifi_ok             # 检查项：集成天线 Wi-Fi 功能检查；true = 已连接指定 AP，false = 未连接/信号异常


# ==========================
# 4. IMU
# ==========================
bool imu_comm_ok                 # 检查项：IMU 状态检查；true = CCU 能持续收到 IMU 通信数据，false = 通信中断/校验失败


# ==========================
# 5. 摄像头（环视）
# ==========================
bool cam_front_ok                # 检查项：前环视摄像头状态检查；true = 正常出图，false = 无图像/黑屏/花屏
bool cam_rear_ok                 # 检查项：后环视摄像头状态检查；true = 正常出图，false = 无图像/黑屏/花屏
bool cam_left_ok                 # 检查项：左环视摄像头状态检查；true = 正常出图，false = 无图像/黑屏/花屏
bool cam_right_ok                # 检查项：右环视摄像头状态检查；true = 正常出图，false = 无图像/黑屏/花屏


# ==========================
# 6. 激光雷达
# ==========================
bool lidar_front_top_ok          # 检查项：前上激光雷达状态检查；true = 正常输出点云，false = 无点云/异常跳变
bool lidar_front_bottom_ok       # 检查项：前下激光雷达状态检查；true = 正常输出点云，false = 无点云/异常跳变
bool lidar_rear_top_ok           # 检查项：后上激光雷达状态检查；true = 正常输出点云，false = 无点云/异常跳变
bool lidar_rear_bottom_ok        # 检查项：后下激光雷达状态检查；true = 正常输出点云，false = 无点云/异常跳变


# ==========================
# 7. 腿部电机通信
# ==========================
bool motor_lf_ok                 # 检查项：左前腿部电机状态检查；true = 3 个电机通信全部正常，false = 任一电机掉线/超时
bool motor_rf_ok                 # 检查项：右前腿部电机状态检查；true = 3 个电机通信全部正常，false = 任一电机掉线/超时
bool motor_lr_ok                 # 检查项：左后腿部电机状态检查；true = 3 个电机通信全部正常，false = 任一电机掉线/超时
bool motor_rr_ok                 # 检查项：右后腿部电机状态检查；true = 3 个电机通信全部正常，false = 任一电机掉线/超时


# ==========================
# 8. 拓展坞 & 外设
# ==========================
bool dock_comm_ok                # 检查项：拓展坞状态检查；true = 所有通信链路正常，false = 任一链路异常/掉线
bool status_led_ok               # 检查项：状态灯板状态检查（4 个）；true = 全部灯板可通信，false = 任一灯板掉线
bool light_led_ok                # 检查项：照明灯板状态检查（2 个）；true = 全部灯板可通信，false = 任一灯板掉线
bool speaker_ok                  # 检查项：扬声器状态检查；true = 扬声器可通信且可播放测试音，false = 无响应/无声音

