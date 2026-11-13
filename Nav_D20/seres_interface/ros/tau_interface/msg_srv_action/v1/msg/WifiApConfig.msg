uint8 BAND_2G = 0x00
uint8 BAND_5G = 0x01

# AP 热点配置（用户设置的参数）
string ssid            # 热点名称
uint8 band             # 频段，使用 BAND_* 常量（2.4G/5G）
uint8 encryption_type  # 加密类型，AP 仅支持 OPEN 或 WPA2_PSK(WifiConfiguration::ENCRYPTION_*)
string password        # 密码（当加密类型为 OPEN 时可为空）
int32 channel          # 信道号（不支持客户端设置，仅用于信息发布，AP 开启时为实际信道，关闭时为配置值）
