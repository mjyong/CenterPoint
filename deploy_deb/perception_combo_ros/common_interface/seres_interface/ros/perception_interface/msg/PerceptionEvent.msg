uint32 SAFETY_HELMET_AND_PPE_CHECK = 0              # 检测安全帽佩戴情况，袖套，秋夏季工作服，马甲穿戴情况，涂装车间需要穿防尘服
uint32 MOBILE_PHONE_USAGE_DURING_WORK_CHECK = 1     # 检测人员是否在工作期间玩手机，打电话。
uint32 POSTURE_COMPLIANCE_CHECK = 2                 # 检测人员姿态是否合规（是否倚靠设备、半躺等未站直状态）
uint32 FEMALE_EMPLOYEE_LONG_HAIR_CHECK = 3          # 检测女性员工长发是否盘起藏于帽内（放置机械卷入风险）
uint32 PEDESTRIAN_PATHWAY_WALKING_CHECK = 4         # 检测员工是否未在人行道上行走
uint32 WORKSHOP_WALKING_CHECK = 5                   # 检测是否车间内行走不按规定道路行走
uint32 LINE_CROSSING_CHECK = 6                      # 检测员工是否跨越线体
uint32 WALKING_PHONE_CALL_CHECK = 7                 # 检测员工是否行走时打电话
uint32 ITEM_DROP_CHECK = 8                          # 检测是否由物品/料架等掉落到地面
uint32 RACK_PLACEMENT_CHECK = 9                     # 检测料架摆放是否超出划定区域
uint32 ITEMS_SCATTERED_DETECTION = 10               # 物品散落检测
uint32 PERSONNEL_BEHAVIOR_STANDARD_DETECTION = 11   # 人员行为规范检测
uint32 DRESS_CODE_STANDARD_DETECTION = 12           # 着装规范检测
uint32 INFRARED_HIGH_TEMPERATURE_DETECTION = 13     # 红外高温检测
uint32 BUILDING_APPEARANCE_CHANGE_DETECTION = 14    # 建筑物外观变化检测
uint32 STREET_LIGHT_WORKING_STATUS_DETECTION = 15   # 路灯工作状态检测
uint32 SMOKE_DETECTION = 16                         # 抽烟检测

uint32 EVENT_CAR_DOOR_OPEN = 17                     # 车门开启事件
uint32 EVENT_PILE_MATERIAL_BOX_READY =18            # 料箱码垛准备好

uint32 TRUCK_PARKING_DETECTION = 19                 # 货车停放检测



uint32 detect_cycle       #检测周期，单位ms。0:实时检测，非0：按指定周期检测
uint32 event_id           # 待检测事件ID