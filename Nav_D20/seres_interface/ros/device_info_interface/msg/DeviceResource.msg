# DeviceResource.msg

# --- 设备类型枚举定义 (常量) ---
uint8 DEVICE_CCU_SOC = 1
uint8 DEVICE_CCU_MCU = 2
uint8 DEVICE_PMU_MCU = 3
uint8 DEVICE_TMU_SOC = 4
uint8 DEVICE_TMU_MCU = 5


# --- 消息字段 ---
uint8 dev_type             # 参考上方设备类型枚举定义
float32 cpu_usage          # CPU利用率，单位：百分比
float32 cpu_temperature    # CPU温度，单位℃
float32 memory_usage       # 内存利用率，单位：百分比
float32 disk_usage         # 磁盘利用率，单位：百分比
ProcessInfo[] process_info # 各进程信息
float32 gpu_usage          # GPU利用率，单位：百分比
float32 npu_usage          # NPU利用率，单位：百分比