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

# SIM卡状态(枚举型)
uint8 SIM_CARD_STATUS_NONE=0x01                             # Card state unknown
uint8 SIM_CARD_STATUS_ABSENT=0x02                           # Card is absent
uint8 SIM_CARD_STATUS_PRESENT=0x03                          # Card is present
uint8 SIM_CARD_STATUS_ERROR_UNKNOWN=0x04                    # Unknown error state
uint8 SIM_CARD_STATUS_ERROR_POWER_DOWN=0x05                 # Power down
uint8 SIM_CARD_STATUS_ERROR_POLL_ERROR=0x06                 # Poll error
uint8 SIM_CARD_STATUS_ERROR_NO_ATR_RECEIVED=0x07            # Failed to receive an answer to reset
uint8 SIM_CARD_STATUS_ERROR_VOLT_MISMATCH=0x08              # Voltage mismatch
uint8 SIM_CARD_STATUS_ERROR_PARITY_ERROR=0x09               # Parity error
uint8 SIM_CARD_STATUS_ERROR_SIM_TECHNICAL_PROBLEMS=0x0a     # Card returned technical problems

uint8 id                                                    # RHR事件ID[必须使用 RHR_EVENT_* 常量]
uint8 status                                                # SIM卡状态[必须使用 SIM_CARD_STATUS_* 常量]