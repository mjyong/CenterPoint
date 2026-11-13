uint8  MOTION_STA_REJECTED = 1        # 动作被拒绝
uint8  MOTION_STA_DOING = 2           # 动作切换过程中/持续移动中
uint8  MOTION_STA_DONE = 3            # 动作执行完成
uint8  MOTION_STA_ABORTED = 4         # 动作执行异常中止

uint32 motion       # 详见MotionSkill.msg定义 MOTION_*
uint8 state         # motion对应的状态，见MOTION_STA_*