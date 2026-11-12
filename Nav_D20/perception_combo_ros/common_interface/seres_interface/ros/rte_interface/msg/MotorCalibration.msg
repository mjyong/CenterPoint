# 请求部分：
uint8 type                      # 0: 整机标零，1：单电机标零，2：kp阈值标定，3：kd阈值标定
uint8 JOINT_CAL_TYPE_ALL_MARKED_ZERO = 0    # 关节全关机标零
uint8 JOINT_CAL_TYPE_SINGLE_MARKED_ZERO = 1 # 单关节标零
uint8 JOINT_CAL_TYPE_KP_THRESHOLD = 2       # kp阈值标定
uint8 JOINT_CAL_TYPE_KD_THRESHOLD = 3       # kd阈值标定
MotorCalibrationParam[] param   # 电机标定参数列表（整机标零时可不填写，其他模式需填写）