int32 label
string class_name
string task # det or seg
float32 confidence
float32[] coor # [x,y,x,y,x,y.....] 检测的结果是左上右下两个点的坐标，分割是边界点的坐标