# 日志信息
std_msgs/Header header       # ros2标准消息头
uint8 log_source               # 日志源 1：CCU-MCU,2：PMU
uint8 log_level                # 日志等级 DEBUG(0),INFO（1）,WARNING（2）,ERROR （3）,FATAL（4）
string log_content             # 具体的日志内容
uint8 log_color                # 日志颜色 DEBUG(常规日志)：常规日志,INFO（信息）：通常用绿色，WARNING（警告）：一般为黄色，ERROR（错误）：常显示为红色，FATAL（致命）：多为深红色或者带有加粗、高亮等样式