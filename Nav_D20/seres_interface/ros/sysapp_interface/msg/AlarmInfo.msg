uint32 id           # 详见告警 ID 列表
uint32 para         # 告警参数,(额外参数,可用于电机id、相机id等)
uint64 time         # 告警状态变化时间，产生或消除 系统时间
uint8 state         # 0:告警产生,1:告警消除
uint8 level         #  WARNING
string comment      #告警描述-必须带内容