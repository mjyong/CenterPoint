# 操作类型(枚举型)
uint8 HANDLE_TYPE_NONE=0x00         # 未定义
uint8 HANDLE_TYPE_ADD=0x01          # 增加
uint8 HANDLE_TYPE_CANCLE=0x02       # 删除/清理
uint8 HANDLE_TYPE_UPDATE=0x03       # 修改
uint8 HANDLE_TYPE_GET=0x04          # 查询/获取

uint8 handle_type                   # 操作类型[必须使用 HANDLE_TYPE_* 常量]