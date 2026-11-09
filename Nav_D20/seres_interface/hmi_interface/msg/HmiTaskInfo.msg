# priority取值参考 common_interface/ResourcePriority.msg 中的 PRIORITY_*
common_interface/ResourcePriority  priority     # 业务优先级1，3，5，7
uint16 index                                    # 全局优先级
string effect                                   # 效果映射枚举
string request_id                               # 请求ID
string request_name                             # 调用方名称（用于日志/审计）
string last_update_time                         # 最后更新时间
