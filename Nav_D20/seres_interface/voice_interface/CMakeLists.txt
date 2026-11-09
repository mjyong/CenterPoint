cmake_minimum_required(VERSION 3.8)
project(voice_interface)

# 设置编译标准
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译器选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

# ==================== ROS接口 ====================
set(msg_files
  "msg/VoiceMessage.msg"
  "msg/VoiceContent.msg"
  "msg/RtAsrAudioData.msg"
  "msg/RtAsrResult.msg" 
)

set(srv_files
  "srv/VoiceService.srv"
  "srv/RtAsrOperation.srv"
)

# 生成自定义接口文件
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES 
    builtin_interfaces 
    std_msgs 
    action_msgs
  ADD_LINTER_TESTS
)

# 导出依赖
ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  std_msgs
)

# 测试相关
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# 导出包含目录
ament_export_include_directories(include)
ament_package()
