string name     # 传感器名称， 如 "gnss"
string type       # 传感器类型，如 "gnss"
string frame_id     # ，如 "gnss_link"
string reference_frame     # ，如 "imu_link"
float64[3] translation       # 平移 [x, y, z]
float64[4] rotation        # 旋转（四元数 [x, y, z, w]）
