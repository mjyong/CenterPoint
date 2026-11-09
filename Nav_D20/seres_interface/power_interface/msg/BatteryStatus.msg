# 电池健康状态 (0~100, 单位 %)
float32 soh

# 电池剩余电量 (0~100, 单位 %)
float32 soc

# 电池电压 (0~100, 单位 v)
float32 voltage

# 电池电流 (0~100, 单位 mA)
float32 current

# 剩余工作时间 (单位：秒) 若无法预估，可设为 -1
float32 remaining_time

# 电池温度 (摄氏度)
float32 temperature

# 当前电池充放电状态，例如： "Charging" / "DisCharging" / "Idle"
string charging_state

# 当前充电枪状态，例如： "ChargerConnected" / "ChargerConnected" / "ChargerFault"
string charger_state

# 当前充电枪类型，例如： "ChargingPile" / "WirelessCharging" / "AdapterCharging"
string charger_type

# 电池序列号，唯一编号
string serial_number

