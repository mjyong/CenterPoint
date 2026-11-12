common_interface/Header header
#任务标识
uint32 task_id
#数据实际大小 
uint32 valid_size
#规划任务列表
PlanningSchedule[] planning_sche
#动作任务列表
ActionSchedule[] act_sche
#加载任务文件
string file_path
#加载数据类型
uint8 load_poi_type