std_msgs/Header header
common_interface/GeograpicPose pose_geograpic        # 地理位置
common_interface/GeometryPose pose_geometry          # 地图位置
common_interface/GeometryPose pose_odometry          # 里程计位置，uuid无效为null

geometry_msgs/Twist twist                         #速度 