std_msgs/Header header

uint8 uwb_key_work_status   #0x00: Idle, 0x01: UWB work only, 0x02 BLE work only, 0x03: UWB && BLE, 0x04: 低电量，0x08: 关机
uint8 shutdown_reason       #仅关机模式有效， 0x01:手动按键关机， 0x02:长时间未动作关机，0x03:低电量关机，
uint8 charge_status         #0x00:idle, 0x01:charging, 0x02:charge completed
uint16 batt_voltage         #遥控器电池电压值,单位mV
uint64 rx_package_bytes     #收包统计
uint64 tx_package_bytes     #发包统计
uint8 uwb_ranging           #uwb 测距状态