# event_id 枚举定义
uint8 EVENT_ID_COLLISION=1
uint8 EVENT_ID_FALLDOWN=2
uint8 EVENT_ID_EMERSTOP=3
uint8 EVENT_ID_DISABILITY=4
uint8 EVENT_ID_FREEZE=5

# state 事件状态枚举定义
uint8 STATE_START=0 #产生事件
uint8 STATE_END=1 #消除事件
#消息内容
uint8 event_id #事件id
uint8 state #事件状态
uint64 time  #事件产生时间，系统时间毫秒
