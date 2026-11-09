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

# 信号强度(枚举型)
uint8 NW_SIGNAL_STRENGTH_LEVEL_NONE=0x00                    # 无信号
uint8 NW_SIGNAL_STRENGTH_LEVEL_POOR=0x01                    # 信号差
uint8 NW_SIGNAL_STRENGTH_LEVEL_MODERATE=0x02                # 信号一般
uint8 NW_SIGNAL_STRENGTH_LEVEL_GOOD=0x03                    # 信号良
uint8 NW_SIGNAL_STRENGTH_LEVEL_GREAT=0x04                   # 信号优

uint8 id                                                    # RHR事件ID[必须使用 RHR_EVENT_* 常量]
uint8 strength                                              # 信号强度[必须使用 NW_SIGNAL_STRENGTH_LEVEL_* 常量]