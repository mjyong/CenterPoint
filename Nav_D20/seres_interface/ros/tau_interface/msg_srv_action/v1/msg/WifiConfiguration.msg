# WiFi 加密方式枚举
uint8 ENCRYPTION_OPEN = 0x00
uint8 ENCRYPTION_WPA_PSK = 0x01
uint8 ENCRYPTION_WPA2_PSK = 0x02
uint8 ENCRYPTION_WPA3_SAE = 0x03

# WiFi 网络配置（用于添加/保存网络）
string ssid            # 网络名称
string bssid           # 可选，指定BSSID（MAC地址）
string password        # 密码（开放网络可为空）
uint8 encryption_type  # 加密类型，使用 ENCRYPTION_* 常量
bool is_hidden         # 是否为隐藏网络
int32 priority         # 连接优先级（0-255，数值越大优先级越高，默认值设为 0。备用，不作为 HMI 配置开放）
int32 network_id       # 网络 id，未保存网络需要初始化为 -1. 添加网络后会分配有效 id (大于或等于 0)
IpSettings ip_config   # IP 配置
