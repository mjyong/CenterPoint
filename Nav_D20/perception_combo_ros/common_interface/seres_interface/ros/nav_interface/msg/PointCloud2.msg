common_interface/Header header
#点云高度(如果是无序点云则为1)
uint32 height
# 点云宽度(点数=width*height)
uint32 width
# 是否所有点都有效(无NaN/INF)
bool is_dense
# 数据字节序
bool    is_bigendian
# 单个点的字节长度
uint32  point_step
# 一行的字节长度(width*point_step)
uint32  row_step
# 实际点云数据(序列化)
uint8[] data 
# 描述点云中每个点的字段
common_interface/PointField[] fields