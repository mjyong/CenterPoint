std_msgs/Header header
#消息类型
uint8 type
uint8 WIFI_INFO_UNKNOWN = 0
uint8 WIFI_INFO_STA = 1
uint8 WIFI_INFO_AP = 2

#STA开关状态
uint8 sta_switch_state
uint8 WIFI_STA_CLOSE=0x00               # 关闭
uint8 WIFI_STA_OPEN=0x01                # 打开

#STA可连接wifi名称列表
WifiSsidList[] sta_ssid_list
#STA接入热点信息
WifiSsidInfo sta_ssid_info
#AP开关状态
uint8 ap_switch_state
uint8 WIFI_AP_CLOSE=0x00               # 关闭
uint8 WIFI_AP_OPEN=0x01                # 打开

#当前模式
uint8 mode
uint8 WIFI_MODE_UNKNOWN = 0
uint8 WIFI_MODE_STA = 1
uint8 WIFI_MODE_AP = 2

#AP热点设备连接数量
int32 ap_dev_num
