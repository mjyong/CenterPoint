# 资源优先级消息定义
# 用于在ROS节点间传递资源优先级相关信息
# 优先级数值定义（数值越大，优先级越高）
int32 PRIORITY_NORMAL      = 1     # 场景应用优先级
int32 PRIORITY_VOICE       = 3     # 语音多模态优先级
int32 PRIORITY_REMOTE      = 5     # 远端控制优先级
int32 PRIORITY_ALARM       = 7     # 告警优先级（最高）

int32 priority