cmake_minimum_required(VERSION 3.5)
project(perception_interface)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

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
find_package(rosidl_typesupport_c REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(common_interface REQUIRED)
find_package(nav2_interface REQUIRED)


file(GLOB msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  msg/*.msg
)

file(GLOB srv_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  srv/*.srv
)

file(GLOB action_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  action/*.action
)

# 生成自定义消息文件
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES rosidl_typesupport_c builtin_interfaces std_msgs action_msgs geometry_msgs sensor_msgs common_interface nav2_interface 
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()