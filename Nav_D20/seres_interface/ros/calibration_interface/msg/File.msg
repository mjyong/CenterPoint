string name         # 标定数据名称（唯一指定标定设备）
string format       # 文件格式，eg：json,yaml等
uint8[] data
uint32 crc          # data字段校验码，具体校验算法由实际项目指定