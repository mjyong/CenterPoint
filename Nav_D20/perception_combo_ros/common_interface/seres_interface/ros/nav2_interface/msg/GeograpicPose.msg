
# 地理坐标点（WGS84）
int8 STATUS_UNKNOW =  -1        # 无效定位
int8 STATUS_FIX =      0        # 单点定位，一般精度误差>2m
int8 STATUS_SBAS_FIX = 1        # 差分定位，一般精度误差1m
int8 STATUS_GBAS_FIX = 2        # 固定解，精度误差厘米级

int8 status

float64 latitude      # 纬度 [-90, 90]，单位：度
float64 longitude     # 经度 [-180, 180]，单位：度
float64 altitude      # 海拔高度，单位：米（WGS84 椭球或 MSL）

bool orient_enable                #true: 姿态方向约束，true：有效；
geometry_msgs/Quaternion orient   #方向