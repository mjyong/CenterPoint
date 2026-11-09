#自研UWB接口
std_msgs/Header header

int8 ble_rssi              #ble 信号强度，单位 dBm
uint16 cs_distance      #ble 被测距离，单位cm
uint8 ble_valid         #ble 数据有效性

uint8 uwb_radar         # uwb 雷达检测结果
uint16 uwb_angle        #uwb 角度值 单位：弧度/角度
uint16 uwb_distance     # uwb 距离
int16 uwb_rssi          # uwb rssi信号强度  dBM
uint8 uwb_valid         # uwb 数据有效性