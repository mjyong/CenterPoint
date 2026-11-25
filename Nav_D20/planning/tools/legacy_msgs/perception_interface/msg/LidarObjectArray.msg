# 一帧激光扫描的跟踪目标，供下游规划使用。
std_msgs/Header header             # stamp 为扫描结束时刻（定位时钟）；frame_id 为检测坐标系
string world_frame                 # 定位坐标系（LIO 为 camera_init）
geometry_msgs/Pose world_pose      # header 坐标系在 world_frame 下、header.stamp 时刻的位姿
LidarObject[] objects
