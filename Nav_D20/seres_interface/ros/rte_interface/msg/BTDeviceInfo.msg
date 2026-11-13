# BTDeviceInfo.msg:
#蓝牙设备地址
string address
#信号强度，数值越大，信号越好
int32 signal_level
#连接状态
uint8 connect_state
uint8 BT_DISCONNECTED=0     # 蓝牙已断开
uint8 BT_CONNECTED=1        # 蓝牙已连接
