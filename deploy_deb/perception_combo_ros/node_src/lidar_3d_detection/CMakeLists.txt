cmake_minimum_required(VERSION 3.16)
project(lidar_3d_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter> $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(perception_interface REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)
find_package(kygi665_msgs QUIET)

find_package(CUDAToolkit QUIET)
find_path(TENSORRT_INCLUDE NvInfer.h
          HINTS $ENV{TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
                /usr/local/tensorrt/include)
find_library(NVINFER nvinfer
             HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(NOT DEFINED LIDAR3D_WITH_CUDA)
  if(CUDAToolkit_FOUND AND TENSORRT_INCLUDE AND NVINFER)
    set(_lidar3d_cuda_default ON)
  else()
    set(_lidar3d_cuda_default OFF)
  endif()
  set(LIDAR3D_WITH_CUDA ${_lidar3d_cuda_default} CACHE BOOL "GPU pillar preprocessing and TensorRT engines")
endif()
option(LIDAR3D_TESTS "GoogleTest parity tests" OFF)

set(LIDAR3D_SOURCES
  src/common.cpp
  src/geometry.cpp
  src/npz.cpp
  src/config.cpp
  src/detections.cpp
  src/pose_buffer.cpp
  src/preprocess.cpp
  src/localization.cpp
  src/sequence.cpp
  src/pipeline.cpp
  src/detection/model_meta.cpp
  src/detection/pillars.cpp
  src/detection/decode.cpp
  src/detection/filters.cpp
  src/detection/detector.cpp
  src/tracking/imm.cpp
  src/tracking/association.cpp
  src/tracking/tracker.cpp
  src/prediction/imm_predictor.cpp
  src/parallel.cpp
  src/online.cpp
  src/message_log.cpp
  src/replay_writer.cpp
)

if(LIDAR3D_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT TENSORRT_INCLUDE OR NOT NVINFER)
    message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer); set TENSORRT_ROOT or turn LIDAR3D_WITH_CUDA off")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89)
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  list(APPEND LIDAR3D_SOURCES src/cuda/pillars.cu src/cuda/trt_engine.cpp src/cuda/gpu_detector.cpp)
endif()

add_library(lidar3d STATIC ${LIDAR3D_SOURCES})
target_include_directories(lidar3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar3d PUBLIC Eigen3::Eigen yaml-cpp ZLIB::ZLIB Threads::Threads)
if(LIDAR3D_WITH_CUDA)
  target_compile_definitions(lidar3d PUBLIC LIDAR3D_WITH_CUDA=1)
  target_include_directories(lidar3d PUBLIC ${TENSORRT_INCLUDE})
  target_link_libraries(lidar3d PUBLIC CUDA::cudart ${NVINFER})
endif()

set(ROS_DEPS rclcpp rosbag2_cpp std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs perception_interface)
if(kygi665_msgs_FOUND)
  list(APPEND ROS_DEPS kygi665_msgs)
endif()

add_library(lidar3d_ros STATIC src/ros_convert.cpp)
target_include_directories(lidar3d_ros PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar3d_ros PUBLIC lidar3d)
ament_target_dependencies(lidar3d_ros PUBLIC ${ROS_DEPS})
if(kygi665_msgs_FOUND)
  target_compile_definitions(lidar3d_ros PUBLIC LIDAR3D_HAVE_KYGI=1)
endif()

add_executable(lidar_3d_detection src/main.cpp src/lidar_3d_detection.cpp)
target_include_directories(lidar_3d_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar_3d_detection lidar3d_ros glog::glog)
ament_target_dependencies(lidar_3d_detection ${ROS_DEPS})

add_executable(lidar_3d_bag_replay src/bag_replay.cpp)
target_include_directories(lidar_3d_bag_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lidar_3d_bag_replay lidar3d_ros)
ament_target_dependencies(lidar_3d_bag_replay ${ROS_DEPS})

add_executable(lidar_3d_replay tools/lidar_3d_replay.cpp)
target_link_libraries(lidar_3d_replay PRIVATE lidar3d)

install(TARGETS lidar_3d_detection lidar_3d_bag_replay lidar_3d_replay
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION .)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
