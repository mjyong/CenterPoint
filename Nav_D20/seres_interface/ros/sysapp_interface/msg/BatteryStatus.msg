# 电池健康状态 (0~100, 单位 %)
float32 soh

# 电池剩余电量 (0~100, 单位 %)
float32 soc

# 剩余工作时间 (单位：秒) 若无法预估，可设为 -1
float32 remaining_time

# 电池温度 (摄氏度)
float32 temperature

# 当前电池充放电状态，例如： "充电中" / "放电中" / "空闲"
string charge_discharge_state

# 电池序列号，唯一编号
string serial_number