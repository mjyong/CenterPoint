std_msgs/Header header
# 远程拍照结果
# 通道ID:5  - 彩色，6 - 红外
uint32 take_photo_id
# 图像总数
uint32 num
# 全部JPEG图像数据(抓图全部完成后一次性发布)
sensor_msgs/CompressedImage[] images
