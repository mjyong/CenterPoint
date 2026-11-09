# 任务集基础属性

# 任务集ID(唯一) 添加任务后自动生成
int32 id
# 名称
string name
# 任务类型 e.g. task_info.WooshTaskSet
string type
# 路线
string route
# 编号(唯一) 外部赋值, 未赋值则与ID一致
string no
# 所属适配器 Default: woosh
string adapter
# 所需执行器 Default: woosh
string actuator
# 机器人类型
uint32 rtype
# 优先级, 值越大优先级越高
int32 priority
# 指定机器人ID
uint32[] robots
