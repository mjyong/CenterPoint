# WiFi 连接状态
uint8 STATE_DISCONNECTED = 0x00  # 未连接（空闲）
uint8 STATE_CONNECTING = 0x01    # 正在连接中
uint8 STATE_CONNECTED = 0x02     # 已成功连接
uint8 STATE_FAILED = 0x03        # 连接失败

uint8 INTERNET_UNAVAILABLE = 0x00  # 无互联网连接
uint8 INTERNET_AVAILABLE = 0x01    # 有互联网连接

uint8 STANDARD_UNKNOWN = 0x00  # 未知标准
uint8 STANDARD_LEGACY = 0x01   # 传统模式 (802.11a/b/g)
uint8 STANDARD_N = 0x02        # 802.11n (WiFi 4)
uint8 STANDARD_AC = 0x03       # 802.11ac (WiFi 5)
uint8 STANDARD_AX = 0x04       # 802.11ax (WiFi 6/6E)
uint8 STANDARD_AD = 0x05       # 802.11ad (60GHz)
uint8 STANDARD_BE = 0x06       # 802.11be (WiFi 7)

uint8 WIFI_SIGNAL_LEVEL_NONE = 0x00      # 无信号
uint8 WIFI_SIGNAL_LEVEL_POOR = 0x01      # 信号差
uint8 WIFI_SIGNAL_LEVEL_MODERATE = 0x02  # 信号一般
uint8 WIFI_SIGNAL_LEVEL_GOOD = 0x03      # 信号良
uint8 WIFI_SIGNAL_LEVEL_GREAT = 0x04     # 信号优

uint8 state              # 当前连接状态 STATE_*
string ssid              # 当前正在操作或已连接的 SSID（标识网络）
string bssid             # 当前 BSSID
string mac               # 可选，随机 mac
uint8 standard           # Wifi 标准 STANDARD_*
string ipv4_address      # 当前 IPv4 地址（若已分配）
string ipv4_gateway      # IPv4 网关
string ipv4_netmask      # IPv4 子网掩码（点分十进制，如 "255.255.255.0"）
string[] ipv6_addresses  # 当前所有 IPv6 地址列表（可能包含链路本地、全局单播等）
string ipv6_gateway      # IPv6 网关
string[] dns_servers     # DNS 服务器
int32 frequency          # 工作频段（MHz）
int32 channel            # 信道号
int32 rssi               # 信号强度（dBm）
uint8 signal_level       # 信号强度等级 WIFI_SIGNAL_LEVEL_*
uint8 internet_state     # INTERNET_* 互联网连接状态
int32 link_speed         # 链路上行 (tx) 速度 (Mb)
int32 link_speed_rx      # 链路下行 (rx) 速度 (Mb)
uint64 link_tx_bytes     # 自连接建立以来，无线网卡已发送的总数据量（字节）
uint64 link_rx_bytes     # 自连接建立以来，无线网卡已接收的总数据量（字节）
uint64 link_tx_rate      # 上行实时流量速率（字节/秒）
uint64 link_rx_rate      # 下行实时流量速率（字节/秒）
int32 network_id         # 网络 id
uint8 encryption_type    # 加密类型
bool is_scanning         # 是否正在扫描
string error_message     # 错误原因（当 state=FAILED 时填充）
