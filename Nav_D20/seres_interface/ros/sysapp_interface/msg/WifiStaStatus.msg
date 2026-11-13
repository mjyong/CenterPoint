# WiFi-STA热点连接状态(枚举型)
uint8 WIFI_STA_ACCESS_DISCONNECTED=0x00        # 未连接
uint8 WIFI_STA_ACCESS_CONNECTED=0x01           # 已连接
# 互联网连接状态(枚举型)
uint8 WIFI_STA_INTERNET_DISCONNECTED=0x00      # 未连接
uint8 WIFI_STA_INTERNET_CONNECTED=0x01         # 已连接

bool enabled                           # WiFi模块是否开启
string name   		               # WiFi-STA接入热点名称
uint8 connect_status                   # WiFi-STA热点连接状态[必须使用 WIFI_STA_ACCESS_* 常量]
uint8 internet_status                  # WiFi-STA互联网连接状态[使用 WIFI_STA_INTERNET_* 常量]
