
int8 CATEGORY_NOT_SET=0
int8 CATEGORY_RACK_SET=1
int8 CATEGORY_PARKSPOT_SET=2
int8 CATEGORY_CHARGER_SET=3
int8 CATEGORY_PALLET_PLATFORM_SET=4
int8 CATEGORY_LIFT_TROLLEY_SET=5
int8 CATEGORY_TRACTOR_TRAILER_SET=6
int8 CATEGORY_ROLLER_STATION_SET=7
int8 CATEGORY_ARM_WORKTOP_SET=8

# 货架
woosh_map_msgs/StorageRack rack
# 停车位
woosh_map_msgs/StorageParkspot parkspot
# 充电桩
woosh_map_msgs/StorageCharger charger
# 栈板平台
woosh_map_msgs/StoragePalletPlatform pallet_platform
# 举升料车
woosh_map_msgs/StorageLiftTrolley lift_trolley
# 牵引挂车
woosh_map_msgs/StorageTractorTrailer tractor_trailer
# 辊筒桩
woosh_map_msgs/StorageRollerStation roller_station
# 机械臂作业台
woosh_map_msgs/StorageArmWorktop arm_worktop
int8 category_choice  # deprecated
int8 which
