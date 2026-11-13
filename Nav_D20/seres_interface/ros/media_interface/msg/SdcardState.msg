#SD卡状态信息
#Topic: /media/topic/sdcard_state
#QoS：reliable + transient_local + keep_last(1)
uint32 SPEED_LOW  = 1 # 低速
uint32 SPEED_HIGH = 2 # 高速
uint32 TYPE_EXFAT = 1 # exFAT
uint32 TYPE_OTHER = 2 # 非exFAT
    
uint32 speed # SD卡速率：1:低速; 2:高速
uint32 type  # SD卡类型：1:exFAT; 2:非exFAT