# 机器人WIFi请求

# 类型
woosh_robot_msgs/RobotWiFiOrder order
# Order为 kAdd kReconnect 时为json参数, Order为 kForget 时为WiFi的SSID
uint8[] content
# 是否添加后立即连接, Order 为 kAdd 时有效
bool is_connect_now
# kHotspot时有效, TRUE: 打开热点, FALSE: 关闭热点
bool enable
