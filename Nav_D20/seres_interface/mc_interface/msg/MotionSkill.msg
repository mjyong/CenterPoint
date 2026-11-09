# 动作控制码，编码时请使用宏定义，而非魔鬼数字！

uint32 MOTION_UNKNOWN               = 0

#安全
uint32 MOTION_DAMP                  = 1     # 阻尼
uint32 MOTION_JOINT_LOCK            = 2     # 僵停
uint32 MOTION_JOINT_UNLOCK          = 3     # 解除僵停
uint32 MOTION_JOINT_DISABLE         = 4     # 零力矩(失能)
uint32 MOTION_JOINT_ENABLE          = 5     # 使能
uint32 MOTION_JOINT_INIT_POS        = 6     # 初始姿态

#步态:支持移动的参数parm: vx, vy, vyaw；单位：vxy:m/s, vyaw:rad/s
uint32 MOTION_GAIT_WALK             = 20    # 行走 
uint32 MOTION_GAIT_STEP_STILL       = 21    # 踏步
uint32 MOTION_GAIT_RUN              = 22    # 跑步
uint32 MOTION_GAIT_SLOPE            = 23    # 上下坡(复杂地形平衡行走)
uint32 MOTION_GAIT_STAIRS           = 24    # 上下楼梯
uint32 MOTION_GAIT_CRAWL            = 25    # 匍匐
uint32 MOTION_GAIT_STRAIGHTEN_LEG   = 26    # 腿打直
uint32 MOTION_GAIT_DANCE_STEP       = 27    # 滑步
uint32 MOTION_GAIT_HANDSTAND        = 28    # 倒立
uint32 MOTION_GAIT_HIGH_PLATFORM    = 29    # 上高台


#生命态
uint32 MOTION_LIFE_STARTUP          = 50    # 开机动作
uint32 MOTION_LIFE_SIT              = 51    # 坐姿待机
uint32 MOTION_LIFE_CRAWL            = 52    # 匍匐待机
uint32 MOTION_LIFE_STAND            = 53    # 站立待机

uint32 MOTION_BODY_UNFOLD           = 54    # 全身展开
uint32 MOTION_BODY_FOLD             = 55    # 全身折叠
uint32 MOTION_BODY_UP               = 56    # 身体上升
uint32 MOTION_BODY_DOWN             = 57    # 身体下降


#整体技能
uint32 MOTION_RECOVER_STAND         = 100   # 恢复站立(倒地恢复)
uint32 MOTION_STAND                 = 101   # 站立
uint32 MOTION_SIT_DOWN              = 102   # 坐姿
uint32 MOTION_LIE_FRONT             = 103   # 躺姿,腹部向上
uint32 MOTION_LIE_BACK              = 104   # 趴下，躺姿,背部向上
uint32 MOTION_SQUAT                 = 105   # 蹲姿
uint32 MOTION_BODY_POSE             = 106   # 身体态控制(按频率实时控制)，控制频率20Hz；3个参数：vroll,vpitch,vyaw;单位：vrpy:rad/s
uint32 MOTION_EULER_BODY            = 107   # 身体Euler控制（按速度到指定角度，执行loop次相对于0点的来回摆动），7个参数：roll,pitch,yaw，vr,vp,vy， loop；单位：rpy：rad， vrpy：rad/s 
uint32 MOTION_JOINTS_TARGET_POSE    = 108   # 关节目标姿态位置控制；para[]关节目标位置参数，详见具体设备关节id定义

uint32 MOTION_PICKUP_REAR_LEG       = 130   # 抬右后腿
uint32 MOTION_BOUND_FORWARD         = 140   # 向前弹跳
uint32 MOTION_BOUND_BACKWRD         = 141   # 向后弹跳
uint32 MOTION_BOUND_IN_PLACE        = 142   # 原地弹跳

uint32 MOTION_RAISE_HEAD_NOD        = 150   # 昂首点头
uint32 MOTION_RAISE_HEAD_CHEST_UP   = 151   # 昂首挺胸
uint32 MOTION_PLAY_NEW_YEAR_SIT     = 152   # 坐姿作揖/拜年
uint32 MOTION_SWING_BODY            = 153   # 摆动身体

uint32 MOTION_DANCE_DANGEROUS       = 800   # 舞蹈-Dangerous
uint32 MOTION_DANCE_LION            = 801   # 舞蹈-舞狮
uint32 MOTION_DANCE_RHYTHMIC        = 802   # 舞蹈  para[0]:指定不同舞蹈，详见文档具体说明

#手部为主
uint32 MOTION_HAND_RIGHT_SCRATCH_HEAD       = 1000  # 右手挠头
uint32 MOTION_HAND_RIGHT_SCRATCH_HEAD_SIT   = 1001  # 右手挠头_坐
uint32 MOTION_HAND_RIGHT_WAVE               = 1002  # 挥右手
uint32 MOTION_HAND_RIGHT_WAVE_SIT           = 1003  # 挥右手_坐
uint32 MOTION_HAND_RUB                      = 1004  # 搓手
uint32 MOTION_STRETCH                       = 1021  # 伸懒腰
uint32 MOTION_STRETCH_HAND                  = 1022  # 手/前肢伸展
uint32 MOTION_STRETCH_HAND_SWING            = 1023  # 手/前肢伸展&摆动
uint32 MOTION_FLOD_HAND_SWING               = 1024  # 手/前肢折叠&摆动
uint32 MOTION_HAND_STEP                     = 1025  # 前肢小踏步
uint32 MOTION_HAND_ENDPOINT_POSE            = 1026  # 手部末端姿态控制，para[]依次为左手右手位姿数据，共7*2个，位姿数据包含position(x,y,z)+orientation(x,y,z,w)
uint32 MOTION_HAND_WAVE_BYE_SIT             = 1027  # 挥手告别_坐

#耳部为主
uint32 MOTION_EAR_RETRACT_WHOLLY            = 2000  # 双耳完全收起
uint32 MOTION_EAR_RETRACT_HALF              = 2001  # 双耳稳态半收
uint32 MOTION_EAR_RETRACT_RAPID             = 2002  # 双耳快速收起
uint32 MOTION_EAR_RETRACT_SLOWLY            = 2003  # 双耳缓缓下垂
uint32 MOTION_EAR_AIRPLANE                  = 2004  # 双耳向后平贴 (飞机耳)

uint32 MOTION_EAR_ERECT                     = 2010  # 双耳前倾竖立
uint32 MOTION_EAR_ERECT_INDEPENDENT         = 2011  # 单耳独立竖起
uint32 MOTION_EAR_OPEN_GENTLY               = 2012  # 双耳平缓张开

uint32 MOTION_EAR_VIBRATION_CLOSE           = 2020  # 双耳紧贴颤动
uint32 MOTION_EAR_VIBRATION_FORWARD         = 2021  # 双耳前倾微颤
uint32 MOTION_EAR_FLAP_ALTERNATE            = 2022  # 双耳交替转动
uint32 MOTION_EAR_FLAP_ALTERNATE_QUICK      = 2023  # 双耳交替扇动
uint32 MOTION_EAR_RESPIRATORY_RHYTHM        = 2024  # 双耳呼吸律动
uint32 MOTION_EAR_JUMP_RESET                = 2025  # 双耳弹跳复位
uint32 MOTION_EAR_DODGE                     = 2026  # 单耳后撇闪避, para[0]:左耳：1.0；右耳：2.0，其他无效
uint32 MOTION_EAR_INSTANT_REAR_BOUNCE       = 2027  # 双耳瞬间后弹

#头部为主
uint32 MOTION_HEAD_POSE                     = 3000  # 头部姿态控制(按频率实时控制)，控制频率20Hz；3个参数：vroll,vpitch,vyaw；单位：vrpy:rad/s
uint32 MOTION_EULER_HEAD                    = 3001  # 头部Euler控制（按速度到指定角度，执行loop次相对于0点的来回摆动），7个参数：roll,pitch,yaw，vr,vp,vy， loop；单位：rpy：rad， vrpy：rad/s 
uint32 MOTION_HEAD_NODE                     = 3002  # 点头
uint32 MOTION_HEAD_SHAKE                    = 3003  # 摇头