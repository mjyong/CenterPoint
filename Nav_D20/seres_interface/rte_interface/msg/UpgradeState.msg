std_msgs/Header header
#升级状态
uint8 state
uint8 UPGRADE_STATE_IN_PROGRESS = 0 # 升级中
uint8 UPGRADE_STATE_COMPLETED   = 1 # 升级完成
uint8 UPGRADE_STATE_FAILED      = 2 # 升级失败

#升级进度条 百分比   0–100
uint8 progress_percent
