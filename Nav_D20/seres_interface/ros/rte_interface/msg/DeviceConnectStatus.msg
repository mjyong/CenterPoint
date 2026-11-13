std_msgs/Header header
#设备连接状态
uint8 state
uint8 DEVICE_CONNECTED = 1
uint8 DEVICE_DISCONNECTED = 0

#设备类型
uint8 type
uint8 DEVICE_TYPE_UNKNOWN = 0
uint8 DEVICE_TYPE_BT = 1
uint8 DEVICE_TYPE_WIFI_AP = 2