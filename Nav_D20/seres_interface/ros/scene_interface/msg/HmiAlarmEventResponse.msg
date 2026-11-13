# topic:"/scene/topic/alarm_task_down"
# 发布者：hmi
# 订阅者：agent等其他需要屏幕回复的模块
uint64 timestamp    # 时间戳
string task_id      # uuid
uint32 alarm_id     # 告警码
int32  alarm_module # 告警产生源,参考告警sender枚举
bool   confirm      # true:告警已处理, false：告警未处理
