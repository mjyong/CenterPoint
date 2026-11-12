# 几何地图坐标点

int8 STATUS_UNKNOW =  -1        # 无效定位
int8 STATE_MAP_SUCCESS = 0      # 定位成功
int8 STATE_MAP_FAILED = 1       # 定位失败
int8 STATUS_MAP_LOCATING = 2    # 地图定位中

int8 status

string map_uuid                   
geometry_msgs/Point point 
bool orient_enable                #true: 姿态方向约束，true：有效；        
geometry_msgs/Quaternion orient   #方向
