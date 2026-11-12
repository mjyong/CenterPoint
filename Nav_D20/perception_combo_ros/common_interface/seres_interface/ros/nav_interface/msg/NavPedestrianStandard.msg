common_interface/Header header
bool is_specification # 是否有异常行为

#图像的高度，即图像的行数
uint32 height
#图像的宽度，即图像的列数
uint32 width
#图像的编码格式
string encoding
#图像的实际像素数据
uint8[] data

uint8 is_bigendian
uint32 step