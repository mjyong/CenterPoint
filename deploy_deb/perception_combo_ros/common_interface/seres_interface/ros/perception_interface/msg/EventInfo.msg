
builtin_interfaces/Time stamp                   # 检测时间

common_interface/GeograpicPose pose_geograpic   # 地理位置
common_interface/GeometryPose pose_geometry     # 地图位置
float32 distance                                # 目标与本体距离（m）

sensor_msgs/CompressedImage object              # 目标图像
common_interface/BoundingBox2D[] box            # 目标框信息
