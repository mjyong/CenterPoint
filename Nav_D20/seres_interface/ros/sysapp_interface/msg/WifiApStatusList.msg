# WiFi-AP开关状态(枚举型)
uint8 WIFI_AP_CLOSE=0x00                # 关闭
uint8 WIFI_AP_OPEN=0x01                 # 打开

uint8 wifi_ap_switch               	             # WiFi-AP开关状态[必须使用 WIFI_AP_* 常量]
int32 max_clients                                # 最大客户端数量，TAU默认为1
int32 num                                           # WiFi-AP热点连接设备数量
string ssid                                           # 热点名称，如"Robot_AP_001"，长度1-32字符
string password                                 # 热点密码
string security                                    # 加密方式：open/wpa2/wpa3
