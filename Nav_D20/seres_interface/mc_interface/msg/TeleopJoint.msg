# 单个关节的目标指令

uint8 id    # 关节的索引，详见机器人遥操作关节相关定义
string name # 关节名称，详见机器人遥操作关节相关定义

float32 pos # 目标位置 (单位: rad)
float32 vel # 目标速度 (单位: rad/s)