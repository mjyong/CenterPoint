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

# 网络类型(枚举型)
uint8 NW_TYPE_UNKNOWN=0x00                                  # unknown
uint8 NW_TYPE_VOICE=0x01                                    # CS（语音）发生变化
uint8 NW_TYPE_DATA=0x02                                     # PS（数据）发生变化

# 注册状态(枚举型)
uint8 NW_SERVICE_NONE=0x00                                  # 无服务
uint8 NW_SERVICE_LIMITED=0x01                               # 服务受限
uint8 NW_SERVICE_FULL=0x02                                  # 服务正常

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

# 注网失败原因(枚举型)
uint8 NW_IMSI_UNKNOWN_IN_HSS_DENY_REASON=0x02                                   # HSS 中未知 IMSI
uint8 NW_ILLEGAL_UE_DENY_REASON=0x03                                            # 非法用户设备
uint8 NW_IMEI_NOT_ACCEPTED_DENY_REASON=0x05                                     # 非法 IMEI
uint8 NW_ILLEGAL_ME_DENY_REASON=0x06                                            # 非法移动设备
uint8 NW_EPS_SERVICES_NOT_ALLOWED_DENY_REASON=0x07                              # EPS服务不可用
uint8 NW_EPS_SERVICES_AND_NON_EPS_SERVICES_NOT_ALLOWED_DENY_REASON=0x08         # EPS和非EPS服务都不可用
uint8 NW_UE_ID_CANNOT_BE_DERIVED_BY_THE_NETWORK_DENY_REASON=0x09                # 网络无法访问用户设备的ID号
uint8 NW_IMPLICITY_DETACHED_DENY_REASON=0x0a                                    # 默认去附着
uint8 NW_PLMN_NOT_ALLOWED_DENY_REASON=0x0b                                      # 非法PLMN
uint8 NW_TRACKING_AREA_NOT_ALLOWED_DENY_REASON=0x0c                             # 非法跟踪区
uint8 NW_ROAMING_NOT_ALLOWED_IN_THIS_TRACNING_AREA_DENY_REASON=0x0d             # 跟踪区不允许漫游
uint8 NW_EPS_SERVICES_NOT_ALLOWED_IN_THIS_PLMN_DENY_REASON=0x0e                 # LMN内不允许EPS服务
uint8 NW_NO_SUITALBE_CELLS_IN_TRACKING_AREA_DENY_REASON=0x0f                    # 跟踪区内没有适合的小区
uint8 NW_MSC_TEMPORARILY_NOT_REACHABLE_DENY_REASON=0x10                         # MSC暂时不可达
uint8 NW_NETWORK_FAILURE_DENY_REASON=0x11                                       # 网络失败
uint8 NW_CS_DOMAIN_NOT_AVAILABLE_DENY_REASON=0x12                               # CS域不可用
uint8 NW_EMS_FAILURE_DENY_REASON=0x13                                           # EMS失败
uint8 NW_MAC_FAILURE_DENY_REASON=0x14                                           # MAC地址解析失败
uint8 NW_SYNC_FAILURE_DENY_REASON=0x15                                          # 同步信息失败
uint8 NW_CONGESTION_DENY_REASON=0x16                                            # 网络冲突
uint8 NW_UE_SECURITY_CAPABILITIES_MISMATCH_DENY_REASON=0x17                     # UE安全性不足
uint8 NW_SECURITY_MODE_REJECTED_DENY_REASON=0x18                                # 安全模式被拒
uint8 NW_NOT_AUTHORIZED_FOR_THIS_CSG_DENY_REASON=0x19                           # CSG未认证
uint8 NW_NON_EPS_AUTHENTICATION_UNCACCEPTABLE_DENY_REASON=0x1a                  # Non-EPS认证失败
uint8 NW_REQUESTED_SERVICE_OPTION_NOT_AUTHORIZED_IN_THIS_PLMN_DENY_REASON=0x23  # PLMN中请求的服务选项认证失败
uint8 NW_CS_SERVICE_TEMPORARILY_NOT_AVAILABLE_DENY_REASON=0x27                  # CS服务暂不可用
uint8 NW_NO_ESP_BEARER_CONTEXT_ACTIVATED_DENY_REASON=0x28                       # ESP载波未被激活
uint8 NW_SEMAMTICALLY_INCORRECT_MESSAGE_DENY_REASON=0x5f                        # 语义错误
uint8 NW_INVALID_MANDATORY_INFORMATION_DENY_REASON=0x60                         # 不合法的必要信息
uint8 NW_MESSAGE_TYPE_NON_EXISTENT_OR_NOT_IMPLEMENTED_DENY_REASON=0x61          # 消息类型不存在或未实现
uint8 NW_MESSAGE_TYPE_NOT_COMPATIBLE_WITH_PROTOCAL_STATE_DENY_REASON=0x62       # 消息类型不兼容当前协议
uint8 NW_INFORMATION_ELEMENT_NON_EXISTENT_OR_NOT_IMPLEMENTED_DENY_REASON=0x63   # 信息元素不存在或未实现
uint8 NW_CONDITIONAL_IE_ERROR_DENY_REASON=0x64                                  # 信息元素错误
uint8 NW_MESSAGE_NOT_COMPATIBLE_WITH_PROTOCAL_STATE_DENY_REASON=0x65            # 消息不兼容当前协议

uint8 id                                                # RHR事件ID[必须使用 RHR_EVENT_* 常量]
uint8 type                                              # 网络类型[必须使用 NW_TYPE_* 常量]
uint8 voice_reg                                         # 语音注册状态[必须使用 NW_SERVICE_* 常量]
uint8 voice_rat                                         # 语音网络制式[必须使用 NW_RADIO_TECH_* 常量]
uint8 voice_reg_fail_reason                             # 语音注网失败原因[必须使用 NW_* 常量]
uint8 data_reg                                          # 数据注册状态
uint8 data_rat                                          # 数据网络制式
uint8 data_reg_fail_reason                              # 数据注网失败原因