uint32 dtc                                      # 故障码(详见诊断调查表定义)
uint32 para                                     # 告警参数(额外参数,可用于电机id、相机id等)
uint64 time                                     # 告警状态变化时间,产生或消除试的系统时间
uint8 state                                     # 0:告警产生,1:告警消除
uint8 level                                     # 告警等级(详见诊断调查表定义DTC等级)
string comment                                  # 告警描述-必须带内容