# IP地址获取方式
uint8 METHOD_DHCP = 0x00
uint8 METHOD_STATIC = 0x01

uint8 method               # DHCP 或 静态
string ipv4_address        # 静态IPv4地址（点分十进制）
string ipv4_netmask        # IPv4子网掩码（点分十进制，如 "255.255.255.0"）
string ipv4_gateway        # IPv4默认网关
string ipv6_address        # 静态IPv6地址（如 "2001:db8::1"）
uint32 ipv6_prefix_length  # IPv6前缀长度（0-128）
string ipv6_gateway        # IPv6默认网关（可包含接口标识，如 "fe80::1%wlan0"）
string[] dns_servers       # DNS 服务器（同时支持IPv4/IPv6地址）
