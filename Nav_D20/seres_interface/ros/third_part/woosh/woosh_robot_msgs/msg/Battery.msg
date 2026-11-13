# 机器人电池信息

# 充电状态
woosh_robot_msgs/BatteryChargeState charge_state
# 电量百分比, 0-100数值, 100表示全满, 0表示没有电量
uint32 power
# 电池详细信息, 包含多电池
woosh_robot_msgs/BatteryDetail[] details
woosh_robot_msgs/BatteryOneOfCommu commu
