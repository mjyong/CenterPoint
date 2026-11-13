# 蓝牙连接状态(枚举型)
uint8 BT_DISCONNECTED=0    # 断开
uint8 BT_CONNECTED=1       # 连接

uint8 connect_state         # 连接状态, 使用 BT_* 常量
string addr                 # 蓝牙设备地址
string name                 # 蓝牙设备名称