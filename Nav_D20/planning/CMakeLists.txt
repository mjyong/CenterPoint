cmake_minimum_required(VERSION 3.20)
project(planning LANGUAGES CUDA C CXX)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../install")

add_definitions(-DPlANNING_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# add_definitions(-DROS2_JAZZY) #for ubuntu 22.04 ros2 jzyzy

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

option(ENABLE_ROS2 "Usr Ros2." ON)
message(STATUS "ENABLE_ROS2: ${ENABLE_ROS2}" )

find_package(octomap REQUIRED)
find_package(Eigen3 REQUIRED) 
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_octomap REQUIRED)
find_package(acados REQUIRED)
find_package(osqp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(g2o REQUIRED)
find_package(Ceres REQUIRED)
# find_package(CUDA REQUIRED)

find_package(Protobuf REQUIRED)
file(GLOB PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/**
)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPISU_LIBRARY nppisu ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIST_LIBRARY nppist ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


if(ENABLE_ROS2)
    add_definitions(-DENABLE_ROS)
    find_package(ament_cmake REQUIRED)
    find_package(grid_map_ros REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(diagnostic_msgs REQUIRED)
    find_package(plotjuggler_msgs REQUIRED)
    find_package(rcl_interfaces REQUIRED)
    find_package(rosbag2_storage REQUIRED)
    find_package(rosbag2_cpp REQUIRED)
    find_package(tf2_msgs REQUIRED)
    find_package(lifecycle_msgs REQUIRED)
    find_package(std_srvs REQUIRED)
    find_package(octomap_msgs REQUIRED)
    find_package(pcl_conversions REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)
    find_package(rclcpp_action REQUIRED)
    find_package(mc_interface REQUIRED)
    find_package(nav2_interface REQUIRED)
    find_package(perception_interface REQUIRED)
    find_package(rte_interface REQUIRED)
    find_package(woosh_robot_msgs REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(tf2_eigen REQUIRED)
    find_package(urdfdom REQUIRED)
    find_package(urdf REQUIRED)
    find_package(kdl_parser REQUIRED)
    find_package(fcl REQUIRED)

    file(GLOB ROS_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/communication/ros_bridge/**.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/communication/ros_debug_bridge/**.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/communication/d20_ptz_bridge/**.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/communication/r11_head_bridge/**.cc)

    list(APPEND ROS_INCLUDE_DIRS 
        ${grid_map_ros_INCLUDE_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${diagnostic_msgs_INCLUDE_DIRS}
        ${plotjuggler_msgs_INCLUDE_DIRS}
        ${rcl_interfaces_INCLUDE_DIRS}
        ${rosbag2_storage_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
        ${lifecycle_msgs_INCLUDE_DIRS}
        ${std_srvs_INCLUDE_DIRS}
        ${octomap_msgs_INCLUDE_DIRS}
        ${pcl_conversions_INCLUDE_DIRS}
        ${tf2_geometry_msgs_INCLUDE_DIRS}
        ${rclcpp_action_INCLUDE_DIRS}
        ${mc_interface_INCLUDE_DIRS}
        ${nav2_interface_INCLUDE_DIRS}
        ${perception_interface_INCLUDE_DIRS}
        ${rte_interface_INCLUDE_DIRS}
        ${woosh_robot_msgs_INCLUDE_DIRS}
        ${tf2_geometry_msgs_DIR}/../../../include/tf2_geometry_msgs
    )
    include_directories(
        ${ROS_INCLUDE_DIRS}
    )
endif(ENABLE_ROS2)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OCTOMAP_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${grid_map_core_INCLUDE_DIRS}
    ${grid_map_octomap_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
    ${OSQP_INCLUDE_DIR}
    ${OSQP_EIGEN_INCLUDE_DIR}
)

file(GLOB PROJETC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm/**/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm/**/**/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm/**/**/**/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm/**/**/**/**/**.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/math/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/util/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/easysocket/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/**.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/communication/**.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/preprocess/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/preprocess/**/**.cc
    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/mission/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/mission/**/**.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/mission/**/**/**.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/entry/**.cc
)

file(GLOB CUDA_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/preprocess/**.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runnable/preprocess/**/**.cu
)

set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

file(GLOB EXTERNAL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/**/*.cc
)

add_executable(${PROJECT_NAME}
        ${PROJETC_SRCS}
        ${CUDA_SRCS}
        ${EXTERNAL_SRCS}
        ${ROS_SRCS}
        ${PROTO_SRCS}
        ${PROTO_HDRS}
)
if(ENABLE_ROS2)
    ament_target_dependencies(
        planning
        "rclcpp"
        "rclcpp_action"
        "mc_interface"
        "nav2_interface"
        "perception_interface"
        "rte_interface"
        "woosh_robot_msgs"
        "nav_msgs"
        "grid_map_core"
        "visualization_msgs"
        "diagnostic_msgs"
        "plotjuggler_msgs"
        "rosbag2_storage"
        "lifecycle_msgs"
        "std_srvs"
        "octomap_msgs"
        "pcl_conversions"
        "rosbag2_cpp"
        "grid_map_ros"
        "tf2_ros"
        "tf2_msgs"
        "urdfdom"
        "urdf"
        "kdl_parser"
        "tf2_eigen"
    )
endif(ENABLE_ROS2)

find_package(OpenMP REQUIRED)
target_compile_options ( ${PROJECT_NAME}
        PRIVATE
        ${OpenMP_CXX_FLAGS}
)

target_include_directories(${PROJECT_NAME}
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/third_party/
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROJECT_NAME}  
    ${PROTOBUF_LIBRARIES}
    glog::glog pthread dl rt acados blasfeo hpipm
    Ceres::ceres
    ${OpenCV_LIBRARIES} 
    ${OpenMP_CXX_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
    ${PCL_LIBRARIES}
    ${grid_map_core_LIBRARIES}
    ${grid_map_octomap_LIBRARIES}
    yaml-cpp
    osqp::osqp
    OsqpEigen::OsqpEigen
    fcl
    ${CUDART_LIBRARY} ${NPPIF_LIBRARY} ${NPPC_LIBRARY} ${NPPISU_LIBRARY} ${NPPIST_LIBRARY} ${CULIBOS}
)
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/robot_factory
    DESTINATION lib/${PROJECT_NAME}
)

file(GLOB THIRD_PARTY_LIBS 
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osqp/lib_arm/*.so
)
install(FILES ${THIRD_PARTY_LIBS}
    DESTINATION lib/${PROJECT_NAME}/lib)

if(ENABLE_ROS2)
    ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env_hooks/planning_root_dir.sh.in")
    ament_package()
endif()
