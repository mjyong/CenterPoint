uint32 MANIP_STA_TASK_EXEC_SUCCESS      = 0         # 任务执行成功
uint32 MANIP_STA_TASK_EXEC_FAILED       = 1         # 任务执行失败
uint32 MANIP_STA_TASK_EXEC_DOING        = 2         # 任务执行中
uint32 MANIP_STA_TASK_EXEC_CANCELED     = 3         # 任务已取消

uint32 MANIP_STA_TARGET_UNREACHABLE     = 1001      # 目标不可达
uint32 MANIP_STA_END_EFFECTOR_NOT_READY = 1002      # 末端执行器未就绪
uint32 MANIP_STA_CAPABILITY_UNSUPPORTED = 1003      # 当前机型不支持该能力
uint32 MANIP_STA_NAV_ERROR              = 1004      # 导航异常
uint32 MANIP_STA_MC_ERROR               = 1005      # 运控异常
uint32 MANIP_STA_PERCEPTION_ERROR       = 1006      # 感知异常
uint32 MANIP_STA_NO_IMAGE               = 1007      # 获取图像异常
uint32 MANIP_STA_SCANNER_ERROR          = 1008      # 扫码器异常
uint32 MANIP_STA_NUM_MISMATCH           = 1009      # 结果数量不匹配