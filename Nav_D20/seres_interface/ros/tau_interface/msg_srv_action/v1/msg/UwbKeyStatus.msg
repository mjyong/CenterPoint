# UWB 钥匙连接状态
uint8 LINK_DISCONNECTED=0    # 断开
uint8 LINK_CONNECTED=1       # 连接

uint8 link_status             # 连接状态[必须使用 LINK_* 常量]
string addr                   # 蓝牙设备地址 "XX:XX:XX:XX:XX:XX"
string name                   # 设备名称
float32 batt_soc              # 电量百分比 (0-100)
uint32 timestamp              # 时间戳(毫秒)
