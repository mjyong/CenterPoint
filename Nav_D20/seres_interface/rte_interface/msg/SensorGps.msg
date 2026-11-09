std_msgs/Header header
#纬度（单位：度，北纬为正，范围：-90.0~90.0）
float64 latitude
#南纬（South）：0 北纬（North）：1
uint8 ns
#经度（单位：度，东经为正，范围：-180.0~180.0）
float64 longitude
#西经（West）：0  东经（East）：1
uint8 we
#海拔高度（单位：米，WGS84椭球面，非实际海拔高度 ）
float32 altitude
#相对高度
float32 relative_height
#地面速度（单位：节）
float32 speed
#角度（地面航向）
float32 angle
#磁偏角（正数方向为磁北东偏， 负数方向为磁北西偏）
float32 md
#定位质量：0=无定位，1=GPS单点定位（SPS），2=差分GPS（DGPS），4=RTK固定解（厘米级精度，需基站支持），5=RTK浮点解（亚米级精度）
uint8 position_quality
#使用的卫星系统（同时可存在多卫星，按位（bit）生效）：0x1=北斗（中国）, 0x2=GPS（美国）, 0x4=GLONASS（俄罗斯）, 0x8=Galileo（欧洲），0x10=QZSS（日本）
uint16 used_systems
#参与定位卫星个数
uint8 satellite_num
#hdop水平精度因子
float32 hdop
#差分时间
uint32 diff_time
#差分ID
uint32 diff_time_id
#行优先位置协方差 矩阵：单位平方米（m²）
float32[9] position_covariance
#协方差类型，0=未知，1=估算，2=对角线已知，3=全矩阵已知
uint8 position_covariance_type