# 标定结果反馈
std_msgs/Header header 
uint8 type                      # 0: 整机标零，1：单电机标零，2：kp阈值标定，3：kd阈值标定（反馈结论）
bool status                     # true: 成功，false: 失败
MotorCalibrationParam[] params   # 标定结果

# type
# 0: 整机标零（人形标识为上肢或者下肢，四肢：标识耳朵/头部/四足）
# 1：单电机标零（根据电机序号识别）
# 2：kp阈值（标定）
# 3：kd阈值（标定）
# param： 电机标定参数列表（整机标零时可不填写，其他模式需填写）