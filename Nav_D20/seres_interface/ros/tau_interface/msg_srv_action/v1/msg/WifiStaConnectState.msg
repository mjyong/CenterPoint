# WiFi信号强度(枚举型)
uint8 WIFI_SIGNAL_STRENGTH_LEVEL_NONE=0x00          # 无信号
uint8 WIFI_SIGNAL_STRENGTH_LEVEL_POOR=0x01          # 信号差
uint8 WIFI_SIGNAL_STRENGTH_LEVEL_MODERATE=0x02      # 信号一般
uint8 WIFI_SIGNAL_STRENGTH_LEVEL_GOOD=0x03          # 信号良
uint8 WIFI_SIGNAL_STRENGTH_LEVEL_GREAT=0x04         # 信号优

# WiFi-STA热点连接状态(枚举型)
uint8 WIFI_STA_ACCESS_DISCONNECT=0x00       # 连接
uint8 WIFI_STA_ACCESS_CONNECT=0x01          # 连接

# 数据连接状态(枚举型)
uint8 WIFI_STA_STATUS_NO_DATA=0x00          # 无数据
uint8 WIFI_STA_STATUS_HAS_DATA=0x01         # 有数据

# 互联网连接状态(枚举型)
uint8 WIFI_STA_INTERNET_DISCONNECT=0x00     # 连接
uint8 WIFI_STA_INTERNET_CONNECT=0x01        # 连接

string name                                 # WiFi-STA接入热点名称
uint8 strength                              # WiFi信号强度[必须使用 WIFI_SIGNAL_STRENGTH_LEVEL_* 常量]
uint8 connect_status                        # WiFi-STA热点连接状态[必须使用 WIFI_STA_ACCESS_* 常量]
uint8 upload_status                         # WiFi-STA上行数据状态[必须使用 WIFI_STA_STATUS_* 常量]
uint8 download_status                       # WiFi-STA下行数据状态[必须使用 WIFI_STA_STATUS_* 常量]
uint8 internet_status                       # WiFi-STA互联网连接状态