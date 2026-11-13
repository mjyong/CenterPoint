#相机存储类型
uint8 MEMORY_SD_CARD  = 1      # SD卡
uint8 MEMORY_INTERNAL = 2      # 内部

uint8 memory_type              # 参考相机存储类型定义
float32 memory_total           # 总存储空间，单位：MB
float32 memory_used            # 已使用存储空间，单位：MB
float32 memory_free            # 剩余存储空间，单位：MB