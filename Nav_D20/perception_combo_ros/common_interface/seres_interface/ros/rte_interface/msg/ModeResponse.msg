std_msgs/Header header
#require 每条消息模式数据必须要发送
uint8 mode
uint8 MODE_IDLE = 0
uint8 MODE_WORK = 1
uint8 MODE_FACTORY = 2
uint8 MODE_CALIBRATION =3
uint8 MODE_UPDATE = 4
uint8 MODE_SLEEP = 5
uint8 MODE_ERROR = 6

uint8 result