# RHR事件ID(枚举型)
uint8 RHR_EVENT_NONE=0x00                                   # 未定义
uint8 RHR_EVENT_SIM_CARD=0x01                               # SIM卡事件
uint8 RHR_EVENT_NW_REG_STATUS=0x02                          # 注网事件
uint8 RHR_EVENT_NW_SIGNAL_STRENGTH=0x03                     # 信号强度事件
uint8 RHR_EVENT_DM_SLEEP=0x04                               # 休眠事件
uint8 RHR_EVENT_DM_WAKEUP=0x05                              # 唤醒事件
uint8 RHR_EVENT_TIME_ADJUST=0x06                            # 时间校准事件
uint8 RHR_EVENT_DATA_CALL_HANDLE=0x07                       # 数据拨号操作事件
uint8 RHR_EVENT_VOICE_CALL_HANDLE=0x08                      # 语音拨号操作事件
uint8 RHR_EVENT_SMS_HANDLE=0x09                             # 短信操作事件
uint8 RHR_EVENT_DM=0x0a                                     # 设备事件
uint8 RHR_EVENT_MSD=0x0b                                    # MSD事件
uint8 RHR_EVENT_DATA_CALL_RESULT=0x0c                       # 数据拨号结果事件

# 数据拨号操作类型(枚举型)
uint8 DATA_CALL_CMD_DISCONNECT=0x00                         # 断开
uint8 DATA_CALL_CMD_CONNECT=0x01                            # 连接

# IP类型(枚举型)
uint8 IP_TYPE_UNKNOWN=0x00                                  # 未知
uint8 IP_TYPE_4=0x04                                        # IPv4
uint8 IP_TYPE_6=0x06                                        # IPv6
uint8 IP_TYPE_4_6=0x0a                                      # IPv4+IPv6

# APN类型(枚举型)
uint8 APN_ID_PRIVATE=0x00                                   # APN1，厂家流量
uint8 APN_ID_PUBLIC=0x01                                    # APN2，用户流量

uint8 id                                                    # RHR事件ID[必须使用 RHR_EVENT_* 常量]
uint8 cmd                                                   # 数据拨号操作类型[必须使用 DATA_CALL_CMD_* 常量]
uint8 ip_type                                               # IP类型[必须使用 IP_TYPE_* 常量]
string apn_name                                             # APN名称
uint8 apn_id                                                # APN类型
int32 fail_type                                             # 失败类型
int32 fail_reason                                           # 失败原因