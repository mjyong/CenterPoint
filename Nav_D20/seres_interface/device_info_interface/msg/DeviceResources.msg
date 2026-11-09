# DeviceResources.msg

std_msgs/Header header          # ROS2标准头
DeviceResource[] dev_resource   # 设备资源列表 (注意这里引用了上面的文件名)
CameraMemory[] cam_memory       # 相机存储列表