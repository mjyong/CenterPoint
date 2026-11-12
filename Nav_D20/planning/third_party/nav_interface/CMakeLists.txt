cmake_minimum_required(VERSION 3.8)
project(nav_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

#自定义消息接口添加部分
rosidl_generate_interfaces(${PROJECT_NAME}
  "${PROJECT_SOURCE_DIR}:msg/FollowTarget.msg"
  DEPENDENCIES rosidl_typesupport_c builtin_interfaces std_msgs action_msgs geometry_msgs sensor_msgs std_srvs
)

ament_package()
