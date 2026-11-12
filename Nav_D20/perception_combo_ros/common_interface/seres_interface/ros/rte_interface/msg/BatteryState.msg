std_msgs/Header header           # ROS2 标准消息头

uint32 sw_version                # 软件版本
uint16 hw_version                # 硬件版本
string serial_number             # 电池序列号（保留）
uint16 total_voltage             # 总电压，单位10 mV
int16 current                    # 电流，单位10 mA，充电为负，放电为正
uint16 remaining_capacity        # 剩余容量，单位10 mAh
uint16 nominal_capacity          # 标称容量，单位10 mAh
uint16 number_cycles             # 循环次数（寿命）
uint16 equilibrium_state         # 均衡状态，每 bit 表示一串电芯均衡（1~16 串）
uint32 protection_status         # 保护状态（bit0~bit26 见文档说明）
uint16 soh                       # 电池健康状态百分比，单位 0.1%
uint16 rsoc                      # 剩余容量百分比，单位 0.1%
uint8 charge_state               # 充放电状态，bit0为充电状态，bit1为放电状态，bit2为充电器连接状态，bit3为充电MOS状态，bit4为放电MOS状态
uint16 realtime_power            # 实时功率 mW
uint16 single_charge_capacity    # 单次充电电量 10mwh（W11不支持）
uint16 rated_current             # 额定电流 10mA
uint16 large_current             # 峰值电流 10mA
uint16 rated_voltage             # 额定电压 10mV
uint8 bat_num                    # 电池串数
uint16[] single_cell_vol_arr     # 单体电压数组 (bat_num 个 uint16)
int8[] temperature_arr          # 温度数组 (6个温度：前三个为电芯温度，后三个为mos温度)













