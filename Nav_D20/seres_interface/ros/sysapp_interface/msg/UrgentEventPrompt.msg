# event_id 枚举定义
uint8 EVENT_ID_COLLISION=1    #碰撞
uint8 EVENT_ID_FALLDOWN=2  #摔倒
uint8 EVENT_ID_EMERSTOP=3  #紧急停止按钮
uint8 EVENT_ID_DISABILITY=4  #关节失能
uint8 EVENT_ID_FREEZE=5  #僵停

# state 事件状态枚举定义
uint8 STATE_START=0 #产生事件
uint8 STATE_END=1 #消除事件

# 枚举：事件消除源
uint8 CLEAR_SOURCE_EVENT_TRIGGERED_END_BY_SYSTEM=0    # 系统自动消除
uint8 CLEAR_SOURCE_EVENT_TRIGGERED_END_BY_HUMAN=1     # 人为确认消除

int32 sequence              #序列号
uint8 event_id              #事件id
uint8 state                 #事件状态
uint64 time                 #事件产生时间，系统时间毫秒
uint8 clear_source          #事件消除源
string comment              #扩展字段