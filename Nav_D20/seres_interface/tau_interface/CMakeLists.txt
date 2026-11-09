cmake_minimum_required(VERSION 3.8)
project(tau_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

file(GLOB msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  msg_srv_action/v1/msg/*.msg
)

file(GLOB srv_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  msg_srv_action/v1/srv/*.srv
)

# WiFi ADB 接口仅 QCOM_6490 平台编译生效（WirelessProxy 模块特有功能，其他平台无 adbd）
if(NOT "$ENV{ROBOT_TYPE}" STREQUAL "D30")
  list(FILTER srv_files EXCLUDE REGEX "WifiAdb[A-Za-z]+\\.srv$")
  message(STATUS "tau_interface: SOC_PLATFORM=$ENV{SOC_PLATFORM}, exclude WifiAdb*.srv (non-QCOM_6490)")
else()
  message(STATUS "tau_interface: SOC_PLATFORM=QCOM_6490, include WifiAdb*.srv")
endif()

find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
