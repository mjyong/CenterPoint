# topic:"agv/system_state"
# 发布者：agv_controller
# 订阅者：
# AGV系统状态获取

std_msgs/Header header
string agv_connect_state   # AGV连接状态
string sys_state           # 系统状态
string location_state      # 定位状态
string emergency_state     # 急停状态
string emergency_source    # 急停触发源: BUTTON / EDGE / SOFTWARE / UNKNOWN
int32 agv_x_speed          # x方向速度
int32 agv_y_speed          # y方向速度
int32 agv_w_speed          # 旋转速度
string oba_enable_state    # 避障功能开关状态: ENABLED / DISABLED / UNKNOWN
string operation_state     # 操作模式：AUTO / MANUAL / UNKNOWN
int32 location_pose_x      # 定位 x 坐标
int32 location_pose_y      # 定位 y 坐标
int32 location_pose_yaw    # 定位角度 yaw值
uint32 station_id           # AGV 当前所在地图中的站点号，若不在站点则为0
