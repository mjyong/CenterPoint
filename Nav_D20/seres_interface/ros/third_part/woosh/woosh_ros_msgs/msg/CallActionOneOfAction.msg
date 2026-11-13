
int8 ACTION_NOT_SET=0
int8 ACTION_CHARGE_CONTROL_SET=1
int8 ACTION_LIFT_CONTROL_SET=2
int8 ACTION_LIFT_CONTROL2_SET=3
int8 ACTION_STEP_CONTROL_SET=4
int8 ACTION_LIFT_CONTROL3_SET=5
int8 ACTION_MOVE_BASE_SET=6
int8 ACTION_NAV_MODE_SET=7
int8 ACTION_AUTO_DOCK_SET=8
int8 ACTION_ANY_ACTION_SET=9

# 充电
woosh_ros_msgs/ChargeControl charge_control
# 举升机构
woosh_ros_msgs/LiftControl lift_control
# 升降机构2
woosh_ros_msgs/LiftControl2 lift_control2
# 步进
woosh_ros_msgs/StepControl step_control
# 升降机构3
woosh_ros_msgs/LiftControl3 lift_control3
# 基础导航
woosh_ros_msgs/MoveBase move_base
# 导航模式
woosh_ros_msgs/NavigationMode nav_mode
# 自动对接
#
# docking
woosh_ros_msgs/AutoDock auto_dock
# 内部使用
#
# any action
woosh_ros_msgs/AnyAction any_action
int8 action_choice  # deprecated
int8 which
