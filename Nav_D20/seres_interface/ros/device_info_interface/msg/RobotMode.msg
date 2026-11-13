# 整机运行模式定义
uint32 ROBOT_MODE_UNKNOWN     = 0     # 0x00-未知模式
uint32 ROBOT_MODE_FACTORY     = 16    # 0x10-工厂模式
uint32 ROBOT_MODE_CALIBRATION = 17    # 0x11-标定模式
uint32 ROBOT_MODE_WORK        = 32    # 0x20-工作模式

uint32 robot_mode              # 当前机器人运行模式，参考整机运行模式定义