#场景状态值
uint8 RUNNING = 0
uint8 PAUSING = 1
uint8 PAUSED = 2
uint8 CANCELING = 3
uint8 CANCELED = 4
uint8 SUCCESS = 5
uint8 FAILURE = 6
uint8 PENDING = 7

string id  #场景id
string name  #场景描述
uint8 state #场景状态
string everything