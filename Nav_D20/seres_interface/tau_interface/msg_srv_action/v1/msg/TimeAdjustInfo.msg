# 时间校准类型(枚举型)
uint8 TIME_ADJUST_TYPE_NONE=0           # 校准类型未知
uint8 TIME_ADJUST_TYPE_RTC=1            # 校准类型为RTC
uint8 TIME_ADJUST_TYPE_NTP=2            # 校准类型为NTP
uint8 TIME_ADJUST_TYPE_GNSS=3           # 校准类型为GNSS

uint8 type                              # 时间校准类型[必须使用 TIME_ADJUST_TYPE_* 常量]
string description                      # 描述信息