cmake_minimum_required(VERSION 3.16)
project(lidar_seg)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
# set(CMAKE_C_COMPILER "/usr/bin/gcc-9")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
# 设置 CUDA 标志
# set(CUDA_NVCC_FLAGS "-O3;-gencode;arch=compute_60,code=sm_60")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive" )

# set(PCL_VERSION 1.10)
# set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.14)  
find_package(PCL 1.10 REQUIRED)
find_package(pcl_conversions REQUIRED)
message(STATUS "=============================")
message(STATUS "PCL_LIBRARY_DIRS ${PCL_LIBRARY_DIRS}")
message(STATUS "PCL_INCLUDE_DIRS ${PCL_INCLUDE_DIRS}")
message(STATUS "pcl_conversions_INCLUDE_DIRS ${pcl_conversions_INCLUDE_DIRS}")
message(STATUS "=============================")
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(kygi665_msgs REQUIRED)
# find_package(cv_bridge REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(serespilot_msgs REQUIRED)

# glog 日志库
find_package(glog REQUIRED)


set (
  DEPENDENCY_LIST
  rclcpp
  std_msgs
  rosbag2_cpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  kygi665_msgs
  # cv_bridge
  # OpenCV
  pcl_conversions
  # serespilot_msgs
  # pcl_common
  glog
)

# 添加源文件
set(SOURCE_FILES
    src/main.cpp
    src/Wildpatchworkpp.cpp
)

link_directories(${PCL_LIBRARY_DIRS}
                #  ${serespilot_msgs_LIBRARY_DIRS}
                )

add_definitions(${PCL_DEFINITIONS})
include_directories( include
                    ${PCL_INCLUDE_DIRS}
                    # ${serespilot_msgs_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/../../../../../third_party/toml/include
                    ${PROJECT_SOURCE_DIR}/../../../../../third_party/ullib/include
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

ament_target_dependencies(${PROJECT_NAME} ${DEPENDENCY_LIST})

target_include_directories(${PROJECT_NAME} PUBLIC 
                           ${pcl_conversions_INCLUDE_DIRS} 
                           ${PCL_INCLUDE_DIRS}
                           ) 

target_link_libraries(${PROJECT_NAME} 
                     ${PCL_LIBRARIES}
                     glog::glog
                    #  ${serespilot_msgs_LIBRARIES}
                    #  ${PROJECT_SOURCE_DIR}/../../../../../third_party/ullib/lib/libullib.a
                     )

install(TARGETS
${PROJECT_NAME}
DESTINATION lib/${PROJECT_NAME}
)          

install(DIRECTORY config DESTINATION ./)
# install(DIRECTORY ${PROJECT_SOURCE_DIR}/../config DESTINATION ${PROJECT_SOURCE_DIR}/../../../../../output/ros2/perception/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
