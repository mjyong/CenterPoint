# CPU_ID(枚举型)
uint8 TAU_CPU_ID_NONE=0x00          # 未定义
uint8 TAU_CPU_ID_NAD=0x01           # NAD
uint8 TAU_CPU_ID_MCU=0x02           # MCU

# 服务状态(枚举型)
uint8 SERVICE_STATUS_DEAD=0x00      # 服务停止
uint8 SERVICE_STATUS_RUNING=0x01    # 服务运行

uint8 cpu_id                        # CPU_ID[必须使用 TAU_CPU_ID_* 常量]
uint8 status                        # 服务状态[必须使用 SERVICE_STATUS_* 常量]