std_msgs/Header header
uint8 ecu_type
uint8 module
uint8 index           
uint32 upstream_tx
uint32 upstream_rx
uint32 downstream_tx
uint32 downstream_rx

# ecu_type
# DEV_ALL = 0, // CCU-All
# DEV_CCU_SOC = 1, // CCU-SOC
# DEV_CCU_MCU = 2, // CCU-MCU
# DEV_PMU = 3,     // PMU
# DEV_TMU_SOC = 4, // TMU-SOC
# DEV_TMU_MCU = 5, // TMU-MCU
# module
#DEV_MODULE_ALL = 0, // UNKOWN
#DEV_MODULE_ACTUATOR_LEGS = 1, // 四足执行器（足端）
#DEV_MODULE_ACTUATOR_NECK = 2, // 四足执行器(颈部)
#DEV_MODULE_ACTUATOR_EAR = 3,  // 四足执行器(耳朵)
#DEV_MODULE_ACTUATOR_UPPER_LIMBS = 4, // 人形上肢执行器
#DEV_MODULE_ACTUATOR_LOWER_LIMBS = 5, // 人形下肢执行器
#DEV_MODULE_ACTUATOR_XHAND_LIMBS = 6, // 人形手执行器
#DEV_MODULE_ACTUATOR_BOBY_LIMBS = 7, // 全身执行器
# index 链路编号（D30）
# 1 : 左前腿
# 2 ：右前腿
# 3 ：左后腿
# 4 ：右后腿
# 5 ：耳朵
# 6 ：颈部