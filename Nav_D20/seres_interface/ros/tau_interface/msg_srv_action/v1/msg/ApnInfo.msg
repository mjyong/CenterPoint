# APN类型(枚举型)
uint8 APN_ID_PRIVATE=0x00                                   # APN1，厂家流量
uint8 APN_ID_PUBLIC=0x01                                    # APN2，用户流量

# 连接状态(枚举型)
uint8 APN_STATUS_DISCONNECT=0x00                            # 断开
uint8 APN_STATUS_CONNECT=0x01                               # 连接

uint8 apn_id                                                # APN类型[必须使用 APN_ID_* 常量]
uint8 connect_status                                        # 连接状态[必须使用 APN_STATUS_* 常量]
uint32 up_rate                                              # 上行实时流量速率（字节/秒）
uint32 down_rate                                            # 下行实时流量速率（字节/秒）
uint64 up_flow                                              # 上行流量（字节）
uint64 down_flow                                            # 下行流量（字节）