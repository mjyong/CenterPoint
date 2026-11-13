
uint8 fan_id
uint8 wind_speed

#-------------------------
# fan_id:
    # 0x00: 0号风扇
    # 0x01: 1号风扇
    # 0x02: 2号风扇
    # ...
    # 0xFF: 所有风扇
# wind_speed:
    # 风速百分比
    # 范围: 50~100，默认为50%

# D30 fan_id
#00：PMU1号风扇（A/B样）
#01：PMU2号风扇(A/B样)
#02：PMU3号风扇(A/B样)
#03：TMU风扇（A/B样）
#04：后腿球窝轴流散热风扇（B样PMU）
#05：头部涡流散热风扇（B样TAU-MCU）
#06：胸腔涡流散热风扇（B样CCU-MCU）
#0xff：全部
