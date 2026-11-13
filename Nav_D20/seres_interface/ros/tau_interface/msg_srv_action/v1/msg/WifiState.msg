# WiFi-STATE(枚举型)
uint8 WIFI_STATE_FAILED = 0x00     # 错误状态（可能在打开或关闭过程中出错引起）
uint8 WIFI_STATE_DISABLING = 0x01  # 正在关闭
uint8 WIFI_STATE_DISABLED = 0x02   # 已关闭
uint8 WIFI_STATE_ENABLING = 0x03   # 正在打开
uint8 WIFI_STATE_ENABLED = 0x04    # 已打开

uint8 wifi_state  # 使用 WIFI_STATE_* 常量
