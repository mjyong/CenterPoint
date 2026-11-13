# topic:"agv/manual_control_speed"
# 发布者：agv_controller 的客户端
# 订阅者：agv_controller
# 手动控制速度下发接口，适用于高频率速度控制场景
# 底盘运动 100ms/次（SDK 要求连续运动时发送间隔不小于 100ms，建议 Client 端以 100ms 周期发送）
# 首次收到非零速度时 agv_controller 自动进入手动控制模式
# 速度全零或订阅断开（watchdog 超时）时自动退出手动控制模式
# 字段语义与原 service 项目 BasicMotion.action Goal 完全一致
std_msgs/Header header
int32 vspeed         # (底盘运动线速度，单位mm/s。正值前进，负值后退)
int32 rspeed         # (底盘运动角速度，单位(1/1000)rad/s。正值逆时针，负值顺时针)
int32 lateral_speed  # (底盘横向速度，单位mm/s。正值右移，负值左移)
