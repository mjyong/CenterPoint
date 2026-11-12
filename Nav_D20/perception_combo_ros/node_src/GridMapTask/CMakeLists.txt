cmake_minimum_required(VERSION 3.8)
project(grid_map_memory_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 通过检测 CMAKE_SYSTEM_PROCESSOR 或其他标识设置不同的 PCL 版本
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(PCL_VERSION 1.2)
    set(PCL_INCLUDE_DIRS /usr/include/pcl-1.12)  
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(PCL_VERSION 1.10)
    set(PCL_INCLUDE_DIRS /usr/include/pcl-1.10)  
else()
    message(FATAL_ERROR "Unsupported architecture")
endif()



# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 交叉编译场景下 sysroot 可能不带完整 VTK，避免拉取 PCL visualization 组件
# 仅声明本节点实际使用的 PCL 组件。
find_package(PCL ${PCL_VERSION} REQUIRED COMPONENTS
  common
  io
  filters
  search
  kdtree
  segmentation
)
find_package(pcl_conversions REQUIRED)
find_package(octomap REQUIRED)  # 添加这一行
find_package(octomap_msgs REQUIRED)
# find_package(perception_msgs REQUIRED)
# find_package(visualperception_msgs REQUIRED)
find_package(nav_interface REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(freespace_msg REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED) # 新增
find_package(yaml-cpp REQUIRED)
find_package(visualization_msgs REQUIRED)

# glog 日志库
find_package(glog REQUIRED CONFIG)
find_package(tf2_ros REQUIRED)
if(NOT SERES_PERCEPTION_ROOT_DIR)
  get_filename_component(SERES_PERCEPTION_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()
set(_fix_sysroot_ros_exports "${SERES_PERCEPTION_ROOT_DIR}/cmake/fix_sysroot_ros_exports.cmake")
if(EXISTS "${_fix_sysroot_ros_exports}")
  include("${_fix_sysroot_ros_exports}")
  fix_sysroot_ros_exports()
endif()
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)



message(STATUS "Detected PCL version: ${PCL_VERSION}")

message(STATUS "Detected OPENCV version: ${OPENCV_VERSION}")


set (
  DEPENDENCY_LIST
  rclcpp
  std_msgs
  rosbag2_cpp
  sensor_msgs
  geometry_msgs
  octomap_msgs
  octomap  # 添加这一行
  # perception_msgs
  # visualperception_msgs
  nav_interface
  pcl_conversions
  cv_bridge
  OpenCV
  # freespace_msg
  nav_msgs
  message_filters
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)

message(STATUS "---------- now ${CUR_DIR}")


# set(FUSION_SRC_FILES
#   src/grid_map_node.cpp
#   )

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
                    ${PCL_INCLUDE_DIRS}
                    ${geometry_msgs_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 安装配置文件
install(
  DIRECTORY cfg/
  DESTINATION ./cfg
)

add_executable(${PROJECT_NAME} 
               src/grid_map_memory_node.cpp
              #  ${FUSION_SRC_FILES}
               )

ament_target_dependencies(${PROJECT_NAME} ${DEPENDENCY_LIST})
if(EXISTS "${_fix_sysroot_ros_exports}")
  fix_sysroot_ros_exports_for_target(${PROJECT_NAME})
endif()

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} yaml-cpp glog::glog curl)
                     
install(TARGETS
${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()



