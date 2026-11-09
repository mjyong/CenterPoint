# 动作指令控制
std_msgs/Header header
string uuid             # 指令UUID，控制端生命周期内一致
uint8 mode              # 使用Mode.msg定义 MODE_*
uint32 motion           # 使用MotionSkill.msg定义 MOTION_*, 支持任意时刻切换,无需过程中取消
float32[] para          # motion参数，详见MotionSkill说明