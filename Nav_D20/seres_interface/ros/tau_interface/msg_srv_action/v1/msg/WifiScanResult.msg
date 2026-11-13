# 扫描到的 WiFi 网络信息
string ssid            # 网络名称（可能为空，隐藏网络可能不广播）
string bssid           # BSSID
uint32 frequency       # 频段（MHz）
int32 rssi             # 信号强度（dBm）
bool is_saved          # 是否是已保存网络
uint8 encryption_type  # 加密类型，使用 WifiEncryptionType 常量
