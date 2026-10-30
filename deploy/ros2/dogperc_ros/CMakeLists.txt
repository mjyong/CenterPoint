# The ROS 2 node around libdogperc (deploy/cpp, built here from the same sources the parity tests check).
#   colcon build --base-paths deploy/ros2 kygi665_msgs --build-base deploy/ros2/build --install-base deploy/ros2/install \
#       --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87
cmake_minimum_required(VERSION 3.16)
project(dogperc_ros LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(DOGPERC_WITH_CUDA "GPU pillar detector and TensorRT (the node needs it)" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(dog_perception_msgs REQUIRED)
find_package(kygi665_msgs QUIET)            # INSPVAX localization (the kygi robot)

# libdogperc, static, from deploy/cpp next to this package (the real path: colcon may see a symlink)
get_filename_component(HERE "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
set(DOGPERC_SOURCE_DIR "${HERE}/../../cpp" CACHE PATH "deploy/cpp")
set(DOGPERC_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${DOGPERC_SOURCE_DIR} dogperc)

set(ROS_DEPS rclcpp builtin_interfaces std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs dog_perception_msgs)
if(kygi665_msgs_FOUND)
  list(APPEND ROS_DEPS kygi665_msgs)
endif()

add_library(dogperc_ros_convert STATIC src/ros_convert.cpp)
target_include_directories(dogperc_ros_convert PUBLIC src)
target_link_libraries(dogperc_ros_convert PUBLIC dogperc)
ament_target_dependencies(dogperc_ros_convert PUBLIC ${ROS_DEPS})
if(kygi665_msgs_FOUND)
  target_compile_definitions(dogperc_ros_convert PUBLIC DOGPERC_HAVE_KYGI=1)
else()
  message(WARNING "kygi665_msgs not found: no INSPVAX localization (put kygi665_msgs into the colcon base paths)")
endif()

add_executable(dogperc_node src/node.cpp)
target_link_libraries(dogperc_node dogperc_ros_convert)
ament_target_dependencies(dogperc_node tf2_ros)

add_executable(dogperc_bag_replay src/bag_replay.cpp)
target_link_libraries(dogperc_bag_replay dogperc_ros_convert)
ament_target_dependencies(dogperc_bag_replay rosbag2_cpp)

install(TARGETS dogperc_node dogperc_bag_replay DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
