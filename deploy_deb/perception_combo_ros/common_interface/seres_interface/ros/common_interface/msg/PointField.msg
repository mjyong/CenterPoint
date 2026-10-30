# PointField.msg
# 字段名称(x, y, z, intensity等)
string name
# 该字段在点数据中的字节偏移量
uint32 offset
# 数据类型
uint8  datatype
# 该字段的元素数量(通常为1)
uint32 count
