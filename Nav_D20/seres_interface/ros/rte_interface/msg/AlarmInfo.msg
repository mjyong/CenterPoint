uint32 id   # 告警id
uint32 para # 额外参数，可用于电机id、相机id等
uint64 time #  告警状态变化时间，产生或消除 系统时间毫秒
uint8 state # 0:告警产生,1:告警消除
uint8 level # 告警级别 1：Warn、2：Error、3：Fatal 
string comment # 告警描述-必须带内容