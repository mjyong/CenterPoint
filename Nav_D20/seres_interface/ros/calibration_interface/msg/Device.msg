uint8 DEVICE_TYPE_CAMERA        = 1
uint8 DEVICE_TYPE_CAMERA_DEPTH  = 2
uint8 DEVICE_TYPE_LIDAR         = 3
uint8 DEVICE_TYPE_UWB           = 4
uint8 DEVICE_TYPE_IMU           = 5
uint8 DEVICE_TYPE_ULTRASONIC_RADAR = 6  #超声波雷达

uint8 device_type             # 标定的设备类型，如上定义 DEVICE_TYPE_*
uint8 device_index            # 标定设备的序号, 由系统统一指定      