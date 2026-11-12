# Tactile.msg
std_msgs/Header header
bool contact                 # 是否接触
float32 force                # 力值，可选（当前为电容传感器该值为0）
float32 signal_strength      # 触觉信号强度，归一化 0~1
uint8 sensor_id              # 传感器编号
string sensor_info           # 位置说明，例如 "left_shoulder"
string sensor_type           # "capacitive"(电容式传感器), "resistive"（电阻式传感器）, "piezo"（压力式传感器）等

# 预留扩展字段
geometry_msgs/Point position        # 可选（保留），接触点位置（传感器坐标系）
geometry_msgs/Vector3 force_vector  # 可选（保留），力方向和大小

float32 min_force                   # 可选（保留），传感器最小力
float32 max_force                   # 可选（保留），传感器最大力
