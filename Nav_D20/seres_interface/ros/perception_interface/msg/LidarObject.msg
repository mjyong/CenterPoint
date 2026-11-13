# 一帧激光检测跟踪结果中的一个目标，坐标系与 LidarObjectArray.header 一致：
# 重力对齐、航向跟随车体、原点在车体正下方地面上。
uint8 LABEL_VEHICLE = 0
uint8 LABEL_PEDESTRIAN = 1
uint8 LABEL_CYCLIST = 2
uint8 MOTION_STATIC = 0            # 未见运动
uint8 MOTION_MOVING = 1
uint8 MOTION_STOPPED = 2           # 曾经运动、当前静止

uint32 id                          # 航迹 id，目标存在期间保持不变
uint8 label
string label_name                  # vehicle / pedestrian / cyclist
uint8 motion
float32 score                      # 航迹置信度
bool coasting                      # 本帧没有检测，由运动模型外推
float32 age                        # [s] 航迹已存在时间
geometry_msgs/Point position       # 框中心 [m]，z 为离地高度
float64 yaw                        # [rad] 框朝向，x 沿长度方向
geometry_msgs/Vector3 size         # 长、宽、高 [m]
geometry_msgs/Vector3 velocity     # 对地速度 [m/s]，z = 0
float64[16] covariance             # [x, y, vx, vy] 的协方差，按行展开
float32[] prediction_times         # [s] 相对 header.stamp
LidarPredictedPath[] predictions   # 概率从高到低；未做预测时为空
