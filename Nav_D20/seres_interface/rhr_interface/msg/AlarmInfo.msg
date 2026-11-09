# 接口描述：告警接口定义

# 发布: 各业务模块
# 订阅：rhr模块
# topic:/rhr/topic/alarm_info

# 发布: rhr模块
# 订阅：alarm_app模块
# topic:/rhr/topic/alarm_item

# 告警状态枚举定义
uint8 ALARM_ON=0  # 告警开始
uint8 ALARM_OFF=1 # 告警结束

# 告警等级枚举定义
uint8 ALARM_GENERAL=1 # 一般告警
uint8 ALARM_SERIOUS=2 # 严重告警
uint8 ALARM_URGENT=3  # 紧急告警

uint32 id      # 告警id:详见健康管理问卷调查表
uint64 time    # 告警时间:产生或消除时间,毫秒时间戳
uint8  state   # 告警状态:参考枚举定义
uint8  level   # 告警等级:参考枚举定义
uint32 para    # 告警参数:未使用
string comment # 告警描述:必填，参考健康管理问卷调查表
string name    # 告警名:必填，参考健康管理问卷调查表