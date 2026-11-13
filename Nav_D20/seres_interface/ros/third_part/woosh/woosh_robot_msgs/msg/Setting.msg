# 机器人基本设置信息

uint8 IDENTITY_FIELD_SET=2
uint8 SERVER_FIELD_SET=4
uint8 POWER_FIELD_SET=8
uint8 SOUND_FIELD_SET=16
uint8 ALLOW_FIELD_SET=32

# 机器人标识
woosh_robot_msgs/Identity identity
# 连接服务器信息
woosh_robot_msgs/Server server
# 电量配置信息
woosh_robot_msgs/Power power
# 声音设置信息
woosh_robot_msgs/Sound sound
woosh_robot_msgs/SettingAllow allow
woosh_robot_msgs/SettingOneOfCommu commu
uint8 has_field 255
