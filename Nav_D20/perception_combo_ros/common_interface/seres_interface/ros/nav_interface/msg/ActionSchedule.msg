# ActionSchedule.msg
# NONE=0, TAKE_PHOTO=1, LINEAR_SCAN=2, RECHARGING=3
uint8 type
TakePhoto photo
LinearScan scan
common_interface/Point recharging_target_point
bool action_on_patrol  # false：到达POI点后再执行动作；true：移动过程中执行
