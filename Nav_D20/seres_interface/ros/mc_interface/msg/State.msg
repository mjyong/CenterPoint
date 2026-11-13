uint32 FAULT_UNKNOWN            = 0     # 异常未知/链路正常
uint32 FAULT_IMU_UNLINK         = 0x01  # IMU断连
uint32 FAULT_JOINT_LOST_HEAD    = 0x02  # 头部关节链路断连
uint32 FAULT_JOINT_LOST_LEG     = 0x04  # 腿部关节链路断连
uint32 FAULT_JOINT_LOST_EAR     = 0x08  # 耳部关节链路断连
uint32 FAULT_JOINT_LOST_UPPER   = 0x10  # 上肢关节链路断连
uint32 FAULT_JOINT_LOST_LOWER   = 0x20  # 下肢关节链路断连
uint32 FAULT_JOINT_TORQUE_OVER_LIMIT    = 0x40  # 扭矩超限
uint32 FAULT_FALL_DOWN          = 0x0100 # 异常倒地

std_msgs/Header header
string uuid                             # 指令UUID， 控制端生命周期内一致，运控内部主动变更时为空
uint8 mode                              # 使用Mode.msg定义 MODE_*
MotionState[] state                     # 四肢、头、耳等不同共存动作时，数组叠加
uint32 fault                            # [废弃接口，不再使用，后续版本移除]有异常时填充上报（多异常按bit组合），详见FAULT_*，无异常时为空
uint32[] fault_arr                      # 有异常时填充上报, 高16bit为关节ID（0：无关具体关节），低16bit为异常ID：FAULT_*；数据有效性与数组顺序无关
