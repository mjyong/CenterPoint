# Proximity.msg
std_msgs/Header header
bool object_detected         # 是否检测到物体
float32 distance             # 距离 (m)，若无法测量填 -1
float32 signal_strength      # 电容可用信号强度 0~1，超声波/红外可用反射强度
uint8 sensor_id              # 传感器ID，如 1
string sensor_info           # 传感器信息，如 "left_hand_1"
string sensor_type           # 类型，如 "capacitive"（电容式接近觉）, "ultrasonic"（超声式接近觉）, "infrared"（红外式接近觉））
float32 field_of_view        # 可选，红外/超声波才有，单位 rad
float32 min_range            # 可选，最小测距 (m)
float32 max_range            # 可选，最大测距 (m)
