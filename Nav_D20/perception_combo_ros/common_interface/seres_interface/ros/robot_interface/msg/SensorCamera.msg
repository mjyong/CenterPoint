#name:sensor_camera
std_msgs/Header header  #ros2标准信息偷，内含时间戳
uint32 height           #图像的高度,即图像的行数
uint32 width            #图像的宽度,即图像的列数
string encoding         #图像的编码格式,"mono8","bgr8","yuyv","rgb8","mjpeg","h264"等,此相机出厂默认为"yuyv"格式
uint8[] data            #图像的实际像素数据