cmake_minimum_required(VERSION 3.8)
project(agv_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

file(GLOB msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg)
file(GLOB srv_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} srv/*.srv)
file(GLOB action_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} action/*.action)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES action_msgs std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
