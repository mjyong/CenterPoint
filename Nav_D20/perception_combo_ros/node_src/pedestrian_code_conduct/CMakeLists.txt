cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
PROJECT(pedestrian_code_conduct VERSION 1.0.0 LANGUAGES C CXX)

message(WARNING "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}" )
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64" )
message(WARNING "xxxxxxxxxxxxxxxxxxxxxx" )
  set(ON_ORIN ON)
  message(WARNING "ON_ORIN: ${ON_ORIN}" )
else()
  set(ON_ORIN OFF)
  message(WARNING "ON_ORIN: ${ON_ORIN}" )
endif()

# set(ON_ORIN_DEBUG ON)


option(ENABLE_ROS2 "Usr Ros2." ON)
message(STATUS "ENABLE_ROS2: ${ENABLE_ROS2}" )
option(ENABLE_SERES_IPC "Usr Seres ipc." OFF)
message(STATUS "ENABLE_SERES_IPC: ${ENABLE_SERES_IPC}" )

# eigen库和opencv库
find_package(Eigen3 REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED CONFIG)
message(STATUS "Detected OPENCV version: ${OPENCV_VERSION}")



if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 添加NEON支持
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
    # 启用NEON指令集
    message(STATUS "USING NEON -------")
    add_definitions(-mfpu=neon)
endif()

# ros2的配置
if(ENABLE_ROS2)
    add_definitions(-DENABLE_ROS)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    
    find_package(rclcpp_action REQUIRED)
    # 尝试查找nav_interface，但不强制要求
    find_package(nav_interface QUIET)
    find_package(perception_interface QUIET)
    

    # 如果cv_bridge没有提供INCLUDE_DIRS，手动设置
    if(${ON_ORIN_DEBUG})
        set(cv_bridge_INCLUDE_DIRS /opt/ros/humble/include/cv_bridge)
        set(cv_bridge_LIBRARIES /opt/ros/humble/lib/libcv_bridge.so)
    elseif(ON_ORIN)
        set(cv_bridge_INCLUDE_DIRS /l4t/jetson_arm64_sysroot/opt/ros/humble/include/cv_bridge)
        set(cv_bridge_LIBRARIES /l4t/jetson_arm64_sysroot/opt/ros/humble/lib/libcv_bridge.so)
    else()
        set(cv_bridge_INCLUDE_DIRS /opt/ros/humble/include/cv_bridge)
        set(cv_bridge_LIBRARIES /opt/ros/humble/lib/libcv_bridge.so)
    endif()
    message(WARNING "Manually set cv_bridge_INCLUDE_DIRS: ${cv_bridge_INCLUDE_DIRS}")

    
    set(ROS_INCLUDE_DIRS
        ${rclcpp_INCLUDE_DIRS}
        ${rclcpp_action_INCLUDE_DIRS}
        ${std_msgs_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${geometry_msgs_INCLUDE_DIRS}
        ${visualization_msgs_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
    )
    
    set(ROS_LIBRARIES
        ${rclcpp_LIBRARIES}
        ${rclcpp_action_LIBRARIES}
        ${std_msgs_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${geometry_msgs_LIBRARIES}
        ${visualization_msgs_LIBRARIES}
        ${cv_bridge_LIBRARIES}
    )
    
    if(nav_interface_FOUND)
        list(APPEND ROS_INCLUDE_DIRS ${nav_interface_INCLUDE_DIRS})
        list(APPEND ROS_LIBRARIES ${nav_interface_LIBRARIES})
        message(STATUS "nav_interface found and added")
    else()
        message(STATUS "nav_interface not found, skipping")
    endif()

    if(perception_interface_FOUND)
        list(APPEND ROS_INCLUDE_DIRS ${perception_interface_INCLUDE_DIRS})
        list(APPEND ROS_LIBRARIES ${perception_interface_LIBRARIES})
        message(STATUS "perception_interface found and added")
    else()
        message(STATUS "perception_interface not found, skipping")
    endif()
    
    # 打印调试信息
    message(STATUS "ROS_INCLUDE_DIRS: ${ROS_INCLUDE_DIRS}")
endif()

# ipc的配置
if(ENABLE_SERES_IPC)
    add_definitions(-DENABLE_SERES_IPC)
    add_definitions(-DCOM_IPC)
    set(IPC_INCLUDE_DIRS
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/include/
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/include/middleware
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/seres_interface
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/seres_interface/ipc
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/seres_interface/ipc/nav_interface
    )
    file(GLOB SERES_IPC_LIBRARIES
        ${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib/**.so
    )
endif(ENABLE_SERES_IPC)

file(GLOB CPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(REMOVE_ITEM CPPS src/main.cpp)
message(STATUS CPPS = ${CPPS})
list (LENGTH CPPS length)
message(STATUS ***length*** = ${length}) 


include_directories(${CUR_DIR}/include
                    ${OPENCV_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
                    ${ROS_INCLUDE_DIRS}
                    ${IPC_INCLUDE_DIRS})


add_library(${PROJECT_NAME} SHARED ${CPPS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUR_DIR}/include
    ${OPENCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${ROS_INCLUDE_DIRS}        # 这些变量此时应该已定义
    ${IPC_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${ALL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${ROS_LIBRARIES}
    ${SERES_IPC_LIBRARIES}
    glog::glog
)

add_executable(pedestrian_code_conduct_2d
                src/main.cpp)

target_link_libraries(pedestrian_code_conduct_2d 
                      ${PROJECT_NAME} 
                      ${OpenCV_LIBRARIES} 
                      ${ROS_LIBRARIES} 
                      ${SERES_IPC_LIBRARIES}
                      glog::glog )

install(TARGETS
${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)


install(TARGETS
    pedestrian_code_conduct_2d
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config DESTINATION ./)