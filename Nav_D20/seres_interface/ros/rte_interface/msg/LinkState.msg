#Link Status
#comm
uint8 link_5g_status      # 5G 通信状态
uint8 link_wifi_status    # WiFi 通信状态
uint8 link_bt_status      # 蓝牙 (BT) 通信状态
uint8 link_eth_status      # 以太网 通信状态
uint8 link_main_status	    #主通信状态
#actuator
uint8[12] link_actuator_status # 执行器状态
#sensor
uint8[4] link_pressure_sensor_status # 压力传感器状态

#perception
uint8 link_imu_status      # IMU 状态
uint8 link_position_status # 定位状态
uint8 link_depth_cam_status # 深度相机状态
uint8 link_wide_cam_status  # 广角相机状态
uint8 link_lidar_status     # LIDAR 状态

uint8 link_battery_status   # 电池状态
uint8 link_fan_status # 风扇状态
uint8 link_led_status     #LED灯链路异常  


 











