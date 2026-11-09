# topic:"/scene/topic/alarm_task_up"
# 发布者：agent等其他需要给屏幕报警的模块
# 订阅者：hmi
int32 ALARM_SENDER_AGENT=0 # agent告警源
int32 ALARM_SENDER_OTHER=1 # 待定
int32 ALARM_TYPE_TASK=0 # 任务类告警
int32 ALARM_TYPE_DEV=1  # 设备类告警
uint64 timestamp  # 时间戳
string task_id    # uuid跟踪任务(必填)
uint32 alarm_id   # 告警码
string alarm_desc # 告警描述
int32  sender     # 告警源,参考枚举
int32  alarm_type # 告警类型,参考枚举