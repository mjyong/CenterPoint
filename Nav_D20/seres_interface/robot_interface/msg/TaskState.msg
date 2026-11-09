# 任务状态
int32 RUNNING   = 0     # 运行中  
int32 SUCCESS   = 1
int32 FAILURE   = 2
int32 CANCELED  = 3     # 已取消

string name             # 任务名称
string id               # 任务UUID
int32 state             # 任务状态
string message          # 可选任务描述信息
