# 运控限制控制指令
# 发布: alarm_app
# 订阅: robot_service
# topic: /robot_service/topic/motion_limit

# 限制类型
uint8 TYPE_HIGH_TEMP = 1   # 电池放电高温预警

# 控制指令
uint8 CMD_ENTER = 0        # 进入/触发
uint8 CMD_EXIT  = 1        # 退出/解除

uint8 type      # 限制类型, 见TYPE_*
uint8 cmd       # 进入/退出, 见CMD_*
string comment  # 描述, 选填(日志留痕)
