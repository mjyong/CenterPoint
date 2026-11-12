std_msgs/Header header
#开关状态
uint8 switch_state
uint8 WIFI_STA_CLOSE=0x00               # 关闭
uint8 WIFI_STA_OPEN=0x01                # 打开

#附近设备数量
uint8 num
#附近蓝牙设备列表
BTDeviceInfo[] device_list
#连接状态
uint8 connect_state
#蓝牙设备地址
string address
