# 机器人常规信息

uint16 MODEL_DATA_FIELD_SET=4
uint16 VERSION_FIELD_SET=256

# 机器人类型
woosh_robot_msgs/Type type
# 机器人尺寸+自重+载重
woosh_robot_msgs/GeneralModelData model_data
# 模型名称
string urdf_name
# 显示名称
string display_model
# 机器人编号
uint32 serial_number
# 机器人服务号
string service_id
# 驱动方式, 0: 两轮差速, 1: 单舵轮, 2: 四舵轮
uint32 driver_method
woosh_robot_msgs/GeneralVersion version
woosh_robot_msgs/GeneralOneOfCommu commu
uint16 has_field 65535
