std_msgs/Header header
float32 roll  #roll:横滚角 (Roll)，单位：度(°) 或 弧度(rad)；
float32 pitch #pitch: 俯仰角 (Pitch)，单位：度(°) 或 弧度(rad)；
float32 yaw   #yaw:偏航角 (Yaw)，单位：度(°) 或 弧度(rad)；
float32 ax   #x 轴加速度，m/s2, float fax = ax / 100
float32 ay   #y 轴加速度，m/s2, float fay = ay / 100
float32 az   #z 轴加速度，m/s2, float faz = az / 100
float32 gx   #x 轴角速度，rad/s, float fgx = gx / 100
float32 gy   #y 轴角速度，rad/s, float fgy = gy / 100
float32 gz   #z 轴角速度，rad/s, float fgz = gz / 100
uint8 valid   #valid:数据有效性指示；