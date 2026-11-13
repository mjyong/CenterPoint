#LidarExtParam.msg
#ID 是对应JSON 里面的数据项，如 T_lidar_front_up_imu
string id
string name   # 传感器名称，如 "lidar_front_up", lidar_front_down
string type    # 传感器类型，如 "lidar"  
string frame_id      # 和  name 保持一致即可
string reference_frame     # lidar_front_up， lidar_front_down, lidar_back_up,  ...
float64[3] translation     # 平移 [x, y, z]
float64[4] rotation          # 旋转（四元数 [x, y, z, w]）
