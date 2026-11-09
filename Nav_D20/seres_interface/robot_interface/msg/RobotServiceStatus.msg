int32 MODE_UNKOWN = 0                          # 无效
int32 MODE_DEBUG = 1                           # 调试模式
int32 MODE_WBMC = 2                            # 全身运控模式
int32 MODE_IMC = 3                             # 下肢运控模式
int32 MODE_TELEOP = 4                          # 遥操模式,该模式下motionskill无效

int32 STATUS_NONE = 0
int32 STATUS_DAMP = 1                           # 阻尼中
int32 STATUS_STALLING = 2                       # 僵停中
int32 STATUS_FAULT = 3                          # 整机故障中， 完全不能动了的那种;
int32 STATUS_DISABLED = 4                       # 失能中
int32 STATUS_NORMAL = 5                         # 正常状态, 机器人可以自行保持平衡的站姿/坐姿/蹲姿/躺姿
int32 STATUS_READY = 6                          # 表示机器人关节就绪, 但是此时即不是站着也不是其他姿态, 仅表示可以切换到指定姿态

int32 BODY_STATE_NONE = 0                       # 默认姿态, 即机器人是瘫着的
int32 BODY_STATE_STAND = 1                      # 全身站姿
int32 BODY_STATE_SIT = 2                        # 坐姿
int32 BODY_STATE_SQUAT = 3                      # 蹲姿
int32 BODY_STATE_LYING = 4                      # 躺姿
int32 BODY_STATE_MOVING = 5                     # 全身移动中（行走， 导航等都算）
int32 BODY_STATE_MOVING_LOWERBODY = 6           # 下半身移动中（下半身行走， 下半身导航等都算）
int32 BODY_STATE_STAND_LOWERBODY = 7            # 下半身站立
int32 BODY_STATE_LIE_BACK = 8                   # 趴姿（背部向上）

uint8 WORK_MODE_NORMAL = 0                      # 工作模式
uint8 WORK_MODE_OTA = 1                         # OTA升级模式（OTA模块调用enterSafePose(Action::ACTION_OTA)时进入），升级结束后自动切换到工作模式
uint8 WORK_MODE_SLEEP = 2                       # 休眠模式（电源模块调用enterSafePose(Action::ACTION_SLEEP)时进入），电源唤醒后自动切换到工作模式
uint8 WORK_MODE_SHUTDOWN = 3                    # 关机模式（电源模块调用enterSafePose(Action::ACTION_SHUTDOWN)时进入），重新开机或关机失败时自动切换到工作模式
uint8 WORK_MODE_CALIB = 4                       # 标定模式（本地web或者上位机调用enterSafePose(Action::ACTION_CALIB_ENTER)时进入），重新开机或退出标定模式时切换到工作模式

int32 pid                                       # 服务端的pid
uint8 work_mode                                 # 工作模式（正常/OTA升级/休眠/关机/标定）
int32 mode                                      # 运控模式 (调试模式/全身运控模式/下肢运控模式/...)
int32 status                                    # 状态 （运控的状态： 零力据/阻尼/初始位姿/正常状态）
int32 body_state                                # 身体姿态（仅STATUS_NORMAL时有效）
RobotServiceFault[] faults                      # 故障
RobotServiceFocusInfo focus_info                # 焦点信息
bool idle                                       # 是否是空闲状态, 没有做任何任务时（生命态这个动作不会影响这个状态）就是空闲状态
uint64 idle_changed_steady_time                 # 空闲状态发生变化时的steady时间戳，单位毫秒; 方便上层应用判断空闲多久了