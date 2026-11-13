# GNSS定位状态(枚举型)
uint8 GNSS_STATUS_INVALID=0x00              # 关闭
uint8 GNSS_STATUS_VALID=0x01                # 打开

uint8 gnss_status                           # GNSS定位状态[必须使用 GNSS_STATUS_* 常量]
float32 longitude                           # 经度
float32 latitude                            # 纬度
float32 speed                               # 速度
float32 heading                             # 航向
int32 satellite_num                         # 搜星数量
float32 hdop                                # 水平定位精度
uint64 utc_time                             # UTC时间戳
float32 altitude                            # 海拔高速