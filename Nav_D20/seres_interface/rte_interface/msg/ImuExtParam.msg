string name    # 传感器名称， # 传感器名称，如 "imu"
string type      # 传感器类型，如 "imu"
string frame_id       #如 "imu_link" 
string reference_frame      #如 "base_link"
float64[3] translation   # 平移 [x, y, z]
float64[4] rotation      # 旋转（四元数 [x, y, z, w]）

