common_interface/Header header 
common_interface/Time map_load_time
# 地图分辨率（米/像素）  
float32 resolution
# 地图宽度（像素数）  
uint32 width
# 地图高度（像素数）
uint32 height
# 地图层级深度
uint32 depth
 # 地图左下角在世界坐标系中的位姿  
common_interface/Pose origin
# 占用值数组（行优先存储） 
MapData2D[] mapdata
#地图类型 
common_interface/MapType type
