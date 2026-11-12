# PlanningSchedule.msg
int32 valid_size
common_interface/Point[] target_points  #type为auto_routing时，只使用第一个点  //100
#是否在终点调整航向
bool attitude_adjustment_at_end
#终点航向
common_interface/Quaternion pose_at_end

