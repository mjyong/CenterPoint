# 连接状态(枚举型)
uint8 APN_STATUS_DISCONNECT=0x00                            # 断开
uint8 APN_STATUS_CONNECT=0x01                               # 连接
# 网络制式(枚举型)
uint8 NW_RADIO_TECH_TD_SCDMA=0x01                           # TD-SCDMA网络（不支持）
uint8 NW_RADIO_TECH_GSM=0x02                                # GSM网络。仅支持语音
uint8 NW_RADIO_TECH_HSPAP=0x03                              # HSPA+网络
uint8 NW_RADIO_TECH_LTE=0x04                                # LTE网络
uint8 NW_RADIO_TECH_EHRPD=0x05                              # eHRPD网络
uint8 NW_RADIO_TECH_EVDO_B=0x06                             # EVDO_B网络
uint8 NW_RADIO_TECH_HSPA=0x07                               # HSPA网络
uint8 NW_RADIO_TECH_HSUPA=0x08                              # HSUPA网络
uint8 NW_RADIO_TECH_HSDPA=0x09                              # HSDPA网络
uint8 NW_RADIO_TECH_EVDO_A=0x0a                             # EVDO_A网络
uint8 NW_RADIO_TECH_EVDO_0=0x0b                             # EVDO_0网络
uint8 NW_RADIO_TECH_1XRTT=0x0c                              # 1xRTT网络
uint8 NW_RADIO_TECH_IS95B=0x0d                              # IS-95B网络
uint8 NW_RADIO_TECH_IS95A=0x0e                              # IS-95A网络
uint8 NW_RADIO_TECH_UMTS=0x0f                               # UMTS网络
uint8 NW_RADIO_TECH_EDGE=0x10                               # EDGE网络
uint8 NW_RADIO_TECH_GPRS=0x11                               # GPRS网络
uint8 NW_RADIO_TECH_NR5G=0x12                               # 5G-NR网络
uint8 NW_RADIO_TECH_NONE=0x13                               # 未知网络
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
# 信号强度(枚举型)
uint8 NW_SIGNAL_STRENGTH_LEVEL_NONE=0x00                    # 无信号
uint8 NW_SIGNAL_STRENGTH_LEVEL_POOR=0x01                    # 信号差
uint8 NW_SIGNAL_STRENGTH_LEVEL_MODERATE=0x02                # 信号一般
uint8 NW_SIGNAL_STRENGTH_LEVEL_GOOD=0x03                    # 信号良
uint8 NW_SIGNAL_STRENGTH_LEVEL_GREAT=0x04                   # 信号优

uint8 connect_status                           # 连接状态[必须使用 APN_STATUS_* 常量]
bool internet_status  			       #互联网状态 是否可访问互联网
uint8 celluar_rat                              # 网络制式[必须使用 NW_RADIO_TECH_* 常量]
uint8 celluar_sim_status                        # SIM卡状态[必须使用 SIM_CARD_STATUS_* 常量]
string celluar_sim_iccid                         # SIM卡ICCID
string celluar_sim_operator                     # SIM卡运营商
uint8 strength                                # 信号强度[必须使用 NW_SIGNAL_STRENGTH_LEVEL_* 常量]
