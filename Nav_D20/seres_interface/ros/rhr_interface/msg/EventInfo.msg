# 接口描述：事件接口定义

# 发布: 各业务模块
# 订阅：rhr模块
# topic:/rhr/topic/event_info

# 发布: rhr模块
# 订阅：alarm_app模块
# topic:/rhr/topic/event_item

# 事件状态枚举定义
uint8 EVENT_ON=0  # 事件开始
uint8 EVENT_OFF=1 # 事件结束

# 告警等级枚举定义
uint8 EVENT_GENERAL=1 # 一般事件
uint8 EVENT_SERIOUS=2 # 严重事件
uint8 EVENT_URGENT=3  # 紧急事件

uint32 id      # 事件id:详见健康管理问卷调查表
uint64 time    # 事件时间:产生或消除时间,毫秒时间戳
uint8  state   # 事件状态:默认填0
uint8  level   # 事件等级:参考枚举定义
string comment # 事件描述:必填，参考健康管理问卷调查表
string name    # 事件名:必填，参考健康管理问卷调查表