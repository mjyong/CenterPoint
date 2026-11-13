# 触发源(枚举型)
uint8 POWER_MGR_TRIGGER_TYPE_NONE=0x00              # 未定义
uint8 POWER_MGR_TRIGGER_TYPE_MCU_KL15=0x01          # KL15触发
uint8 POWER_MGR_TRIGGER_TYPE_MCU_KL30=0x02          # KL30触发
uint8 POWER_MGR_TRIGGER_TYPE_MCU_CAN=0x03           # CAN管理报文触发
uint8 POWER_MGR_TRIGGER_TYPE_MCU_GSENSOR=0x04       # g-sensor触发
uint8 POWER_MGR_TRIGGER_TYPE_NAD_SMS=0x05           # 短信触发
uint8 POWER_MGR_TRIGGER_TYPE_NAD_VOICE=0x06         # 电话振铃触发

# 类型(枚举型)
uint8 POWER_MGR_MODE_TYPE_NONE=0x00                 # 未定义
uint8 POWER_MGR_MODE_TYPE_SLEEP=0x01                # 正常模式to待机模式，休眠
uint8 POWER_MGR_MODE_TYPE_WAKEUP=0x02               # 待机模式to正常模式，唤醒
uint8 POWER_MGR_MODE_TYPE_START=0x03                # 仓储模式to正常模式，启动
uint8 POWER_MGR_MODE_TYPE_END=0x04                  # 正常模式to仓储模式，停止
uint8 POWER_MGR_MODE_TYPE_MAINTAIN_UP=0x05          # 正常模式to维修模式
uint8 POWER_MGR_MODE_TYPE_MAINTAIN_DOWN=0x06        # 维修模式to正常模式
uint8 POWER_MGR_MODE_TYPE_FACTORY_DOWN=0x07         # 工厂模式to正常模式
uint8 POWER_MGR_MODE_TYPE_SHOW_DOWN=0x08            # 展车模式to正常模式
uint8 POWER_MGR_MODE_TYPE_SHOW_UP=0x09              # 正常模式to展车模式

# 阶段(枚举型)
uint8 POWER_MGR_STAGE_NONE=0x00                     # 未定义
uint8 POWER_MGR_STAGE_ONE=0x01                      # 阶段1，应用层处理
uint8 POWER_MGR_STAGE_TWO=0x02                      # 阶段2，服务层处理


uint8 trigger_src                                   # 触发源[必须使用 POWER_MGR_TRIGGER_TYPE_* 常量]
uint8 type                                          # 类型[必须使用 POWER_MGR_MODE_TYPE_* 常量]
uint8 stage                                         # 阶段[必须使用 POWER_MGR_STAGE_* 常量]