uint8 BT_CTRL_DEVICES_A2DP=0	# a2dp设备
uint8 BT_CTRL_DEVICES_HID=1		# hid host设备
uint8 BT_CTRL_DEVICES_GATTC=2	# GATTC设备
uint8 BT_CTRL_DEVICES_GATTS=3	# GATTS设备

uint64 timestamp    # 时间戳
uint8 device        # 设备类型
uint32 len          # 消息长度
uint8[] data        # 消息内容