#主网类型(枚举)
uint8 NETWORK_TYPE_CELLULAR=0x00   #主网为蜂窝网络
uint8 NETWORK_TYPE_WIFI=0x01	   #主网为wifi网络
uint8 NETWORK_TYPE_NONET=0x02      #没有网络
# 信号强度(枚举型)
uint8 NW_SIGNAL_STRENGTH_LEVEL_NONE=0x00                    #无信号
uint8 NW_SIGNAL_STRENGTH_LEVEL_POOR=0x01                    #信号差
uint8 NW_SIGNAL_STRENGTH_LEVEL_MODERATE=0x02                #信号一般
uint8 NW_SIGNAL_STRENGTH_LEVEL_GOOD=0x03                    #信号良
uint8 NW_SIGNAL_STRENGTH_LEVEL_GREAT=0x04                   #信号优

uint8 main_network_type    #主网类型 使用NETWORK_*
bool main_internet_status  #主网互联网状态 是否可访问互联网
uint8 strength             #信号强度[必须使用 NW_SIGNAL_STRENGTH_LEVEL_* 常量]
float32 tx_throughput      #tx_*上行网络吞吐量   单位Mbps
float32 rx_throughput      #rx_*下行网络吞吐量   单位Mbps
float32 tx_error_rate      #tx_*上行网络错误率   单位%
float32 rx_error_rate      #rx_*下行网络错误率   单位%
float32 tx_drop            #tx_*上行网络丢包率   单位%
float32 rx_drop            #rx_*下行网络丢包率   单位%
float32 latency            #网络延迟     单位ms
float32 jitter             #网络抖动     单位ms
