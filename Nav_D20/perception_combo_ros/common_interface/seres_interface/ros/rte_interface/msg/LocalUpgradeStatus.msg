std_msgs/Header header
#升级状态
uint8 state
uint8 LOCAL_UPGRADE_STATE_UNKNOWN = 0
uint8 LOCAL_UPGRADE_DOING = 1 #升级中
uint8 LOCAL_UPGRADE_DONE = 2 # 升级完成
uint8 LOCAL_UPGRADE_FAILED = 3   #升级失败
#升级进度，百分比
uint8 progress