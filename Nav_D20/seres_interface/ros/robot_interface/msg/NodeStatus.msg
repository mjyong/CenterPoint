string node_name                       # 节点名称（如/camera_driver）
string node_namespace                  # 节点命名空间（如/robot1）
string node_type                       # 节点类型（如perception/planning/control）
string node_version                    # 节点软件版本
float32 cpu_usage                      # 节点CPU使用率百分比 (0.0-100.0)
float32 ram_usage                      # 节点RAM使用率百分比 (0.0-100.0)
float32 disk_read_mbps                 # 节点磁盘读取速度 (MB/s，未检测时为-1.0)
float32 disk_write_mbps                # 节点磁盘写入速度 (MB/s，未检测时为-1.0)
# 网络接口状态数组（嵌套NetworkInterface类型）
NetworkInterface[] network_interfaces  # 每个元素描述一个网络接口
string status_message                  # 节点状态描述信息