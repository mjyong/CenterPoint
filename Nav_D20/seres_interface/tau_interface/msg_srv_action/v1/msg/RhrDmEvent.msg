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

# 设备场景(枚举型)
uint8 DM_SCENARIO_TYPE_POWER_ON=0x00                        # power on
uint8 DM_SCENARIO_TYPE_KEY_REBOOT=0x01                      # restart
uint8 DM_SCENARIO_TYPE_AIRPLANE_ON=0x02                     # airplane mode on
uint8 DM_SCENARIO_TYPE_AIRPLANE_OFF=0x03                    # airplane mode off
uint8 DM_SCENARIO_TYPE_DATA_REBOOT=0x04                     # 网络自愈重启tbox
uint8 DM_SCENARIO_TYPE_USB_REBOOT=0x05                      # usb异常错误软重启
uint8 DM_SCENARIO_TYPE_ETH_REBOOT=0x06                      # 以太网恢复

uint8 id                                                    # RHR事件ID[必须使用 RHR_EVENT_* 常量]
uint8 scenario                                              # 设备场景[必须使用 DM_SCENARIO_TYPE_* 常量]