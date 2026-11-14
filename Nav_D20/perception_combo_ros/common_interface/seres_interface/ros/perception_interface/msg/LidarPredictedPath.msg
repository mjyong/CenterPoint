# 一条预测轨迹，位置序列与 LidarObject.prediction_times 对齐，坐标系与 LidarObjectArray.header 一致。
float32 probability
geometry_msgs/Point[] positions    # z = 0
float64[] covariance               # 每个位置 2x2 的 [x, y] 协方差，按行展开，每个点 4 个数
