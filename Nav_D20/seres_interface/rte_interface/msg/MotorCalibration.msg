# 标定请求
uint8 type                      # 0: 整机标零，1：单电机标零，2：kp阈值标定，3：kd阈值标定
uint8 MOTOR_CALIBRATION_TYPE_ZERO_MACHINE = 0 # 关节全关机标零
uint8 MOTOR_CALIBRATION_TYPE_SINGLE_MOTOR = 1 # 单关节标零
uint8 MOTOR_CALIBRATION_TYPE_KP_THRESHOLD = 2 # kp阈值标定
uint8 MOTOR_CALIBRATION_TYPE_KD_THRESHOLD = 3 # kd阈值标定
MotorCalibrationParam[] param   # 电机标定参数列表（整机标零时可不填写，其他模式需填写）  
# type
# 0: 整机标零（人形标识为上肢或者下肢，四肢：标识耳朵/头部/四足）
# 1：单电机标零（根据电机序号识别）
# 2：kp阈值（标定）
# 3：kd阈值（标定）
 # param： 电机标定参数列表（整机标零时可不填写，其他模式需填写）