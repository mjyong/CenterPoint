# 机器人故障类型

int32 FAULT_TYPE_NONE = 0
int32 FAULT_TYPE_AGV_CHASSIS = 1    # AGV错误: 0-无异常/1-有异常
int32 FAULT_TYPE_LOCATE = 2         # 定位功能是否正常: 0-无异常/1-定位失败/2-定位模块无法使用
int32 FAULT_TYPE_FALL_DOWN = 3      # 运控检测出机器异常倒地; 0无异常/1-检测出倒地

int32 fault_type                    # 故障类型
int32 fault_value                   # 故障的值, 各种故障类型自定义的枚举