uint8 dev_type
uint8 DEV_RESOURCE_ALL = 0
uint8 DEV_RESOURCE_SOC = 1 # CCU-SOC
uint8 DEV_RESOURCE_MCU = 2 # CCU-MCU
uint8 DEV_RESOURCE_PMU = 3 # PMU
float32 cpu_usage             # CPU利用率，单位：百分比
float32 gpu_usage             # GPU利用率，单位：百分比
float32 cpu_temperature       # CPU温度，单位：摄氏度
float32 memory_available      # 内存可用余量，单位：百分比
float32 disk_free_space       # 磁盘剩余空间，单位：百分比

#------------------------------
# dev_type
#DEV_RESOURCE_ALL = 0, // CCU-All
#DEV_RESOURCE_SOC = 1, // CCU-SOC
#DEV_RESOURCE_MCU = 2, // CCU-MCU
#DEV_RESOURCE_PMU = 3, // PMU