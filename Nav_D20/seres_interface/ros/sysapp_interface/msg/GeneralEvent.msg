# event_id 枚举定义
uint32 EVENT_ID_WIFI_CONNECTED=10001        #WiFi连接
uint32 EVENT_ID_WIFI_DISCONNECT=10002       #WiFi断开
uint32 EVENT_ID_BT_CONNECTED=10003          #蓝牙连接
uint32 EVENT_ID_BT_DISCONNECT=10004         #蓝牙断开
uint32 EVENT_ID_ROBOT_STARTUP=10005         #机器人开机
uint32 EVENT_ID_ROBOT_SHUTDOWN=10006        #机器人关机
uint32 EVENT_ID_ROBOT_WAKEUP=10007          #机器人唤醒
uint32 EVENT_ID_ROBOT_SLEEP=10008           #机器人休眠
uint32 EVENT_ID_RETURN_CHARGING=10009       #返回充电
uint32 EVENT_ID_START_CHARGING=10010        #开始充电
uint32 EVENT_ID_CHARGING_COMPLETE=10011     #充电结束
uint32 EVENT_ID_MODE_SWITCH=10012           #模式切换
uint32 EVENT_ID_UPGRADE_START=10013         #升级开始
uint32 EVENT_ID_UPGRADE_SUCCESS=10014       #升级成功

# state 事件状态枚举定义，目前统一由发布者填STATE_START，消除作为后续预留功能。
uint8 STATE_START=0 #产生事件
uint8 STATE_END=1 #消除事件

# level 事件紧急处理程度枚举定义
uint8 URGENT_LEVEL_NONE=1   #紧急处理程度（无）
uint8 URGENT_LEVEL_LOW=2    #紧急处理程度（低）
uint8 URGENT_LEVEL_HIGH=3   #紧急处理程度（高）

#消息内容
uint32 event_id     #事件id
uint8 state         #事件状态，目前统一由发布者填STATE_START，消除作为后续预留功能
uint64 time         #事件产生时间，系统时间，单位：毫秒
uint8 level         #事件紧急处理程度，根据仲裁文档中紧急处理程度填充
