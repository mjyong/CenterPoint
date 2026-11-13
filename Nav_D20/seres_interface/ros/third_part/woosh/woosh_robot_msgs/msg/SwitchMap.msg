# 切换地图请求

# 场景名
string scene_name
# 地图名
string map_name
# TRUE: 切换地图后不跳故障, 但需要调用SetRobotPose复位
bool user_reset
# 为空仅切换, 否则一并更新
woosh_common_msgs/FileData[] file_datas
