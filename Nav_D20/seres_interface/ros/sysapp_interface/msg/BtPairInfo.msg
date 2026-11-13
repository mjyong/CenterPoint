# 信号强度(枚举型)
uint8 BT_SIGNAL_STRENGTH_LEVEL_NONE=0x00                    # 无信号
uint8 BT_SIGNAL_STRENGTH_LEVEL_POOR=0x01                    # 信号差
uint8 BT_SIGNAL_STRENGTH_LEVEL_MODERATE=0x02                # 信号一般
uint8 BT_SIGNAL_STRENGTH_LEVEL_GOOD=0x03                    # 信号良
uint8 BT_SIGNAL_STRENGTH_LEVEL_GREAT=0x04                   # 信号优
# 蓝牙连接状态(枚举型)
uint8 BT_DISCONNECTED=0                                     # 蓝牙已断开
uint8 BT_CONNECTED=1                                        # 蓝牙已连接
# 蓝牙设备类型
uint8 BT_CTRL_DEVICES_A2DP=0   # a2dp设备
uint8 BT_CTRL_DEVICES_HID=1    # hid host设备
uint8 BT_CTRL_DEVICES_GATTC=2  # GATTC设备
uint8 BT_CTRL_DEVICES_GATTS=3  # GATTS设备

string addr                   # 蓝牙设备地址
string name                   # 蓝牙设备名称
uint8 strength                # 信号强度[使用BT_SIGNAL_STRENGTH_LEVEL_*常量]
uint8 connect_status          # 蓝牙连接状态[必须使用BT_*常量]
uint8 device                  # 蓝牙设备类型[必须使用BT_CTRL_DEVICES_*常量]
