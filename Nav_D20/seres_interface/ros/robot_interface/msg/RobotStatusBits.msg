int32 BIT_NONE = 0

# 紧急暂停
int32 BIT1_EMERGENCY_PAUSED = 1
# 状态2 (0b10)
# int32 BIT1_STATUS2 = 2

# bit1标志位, 通过(bit1_mask & BIT1_EMERGENCY_PAUSED) ==  BIT1_EMERGENCY_PAUSED 判断是否包含状态
int32 bit1_mask