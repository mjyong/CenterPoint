uint8 device_type
uint8 power_state #电源状态模式
# 0：下电（停止）模式
# 1：上电（激活）模式
# 2：休眠模式
# 3：急停模式（表示外设停止工作（device_type = 0Xff，且power_state=3表示触发急停动作）
# 4：表示UNKNOWN
