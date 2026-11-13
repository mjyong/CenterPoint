# 传输协议类型(枚举型)
uint8 TRANSPORT_PROTOCOL_TYPE_NONE=0x00     # 未定义
uint8 TRANSPORT_PROTOCOL_TYPE_FTP=0x01      # FTP

uint8 type                                  # 传输协议类型[必须使用 RANSPORT_PROTOCOL_TYPE_* 常量]
int32 status                                # 传输服务状态，0-已启动，1-启动失败，2-未启动，3-启动中