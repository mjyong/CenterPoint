cmake_minimum_required(VERSION 3.8)
project(kygi665_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Kygi665Header.msg"
  "msg/InertialSolutionStatus.msg"
  "msg/PositionOrVelocityType.msg"
  "msg/INSExtendedSolutionStatus.msg"
  "msg/GNSS.msg"
  "msg/ODM.msg"
  "msg/IMUAHRS.msg"
  "msg/INSPVAX.msg"
  DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
rosidl_get_typesupport_target(py_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_py")
ament_package()