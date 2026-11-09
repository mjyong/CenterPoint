uint32 id                   # 目标唯一跟踪ID
string class_name           # 类别标签（如 "person"）详见感知目标类型定义
float32 score               # 置信度 [0..1]

float32 x                   # x、y、z：3Dbox中心点位，相对于本体质心坐标系（右手坐标系，xyz-前左上）
float32 y
float32 z
float32 length              # 3D目标长宽高，单位：m
float32 width
float32 height