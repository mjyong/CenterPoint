cmake_minimum_required(VERSION 3.5)
project(device_info_interface)

# 默认 C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 核心依赖 (只保留生成消息必须的) ---
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# --- 自动遍历 msg 文件夹下所有 .msg 文件 ---
file(GLOB msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "msg/*.msg")

# --- 生成接口 ---
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()