# 对接

uint8 IDENTIFY_FIELD_SET=1
uint8 VERIFY_FIELD_SET=2

woosh_map_msgs/DockIdentify identify
# 校验
woosh_map_msgs/DockVerify verify
# 特征板类型
woosh_map_msgs/DockFeatureBoardType board_type
# 识别tag
string tag
uint8 has_field 255
