common_interface/Header header 
float32 p
float32 t
float32 z
bool ptz_in_place   #静态任务指令云台是否到位
uint8 poi_id        #当前任务点位ID
uint8 task_type     #当前任务类型:1 路灯检测（RGB图像）;2 建筑物外立面检测（RGB图像）;3 行人行为规范（RGB图像）;4 高温检测（红外图像）