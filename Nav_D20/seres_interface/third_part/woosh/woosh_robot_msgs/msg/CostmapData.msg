# 代价地图数据

# 场景名
string scene_name
# 地图名
string map_name
# 模型类型
woosh_robot_msgs/FootPrint type
# 构图中的地图数据, 是一个png图片数据
uint8[] map_data
# 图片的宽
uint32 width
# 图片的高
uint32 height
# 图片左下角x坐标
float32 origin_x
# 图片左下角y坐标
float32 origin_y
# 地图分辨率
float32 resolution
woosh_robot_msgs/CostmapDataOneOfCommu commu
