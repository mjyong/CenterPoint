# TraversabilityGrid.msg
common_interface/Header header 
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
#代价地图
CostMap costmap
#地图类型 
uint8 type
