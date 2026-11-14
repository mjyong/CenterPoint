# lidar_3d_detection: the C++ pipeline (library lidar3d: preprocessing, detection, tracking, prediction; the
# CenterPoint repository's dog_perception in C++), the ROS 2 node, the bag replay, the replay tool, and the parity
# tests against the Python reference. Two ways to build it:
#
# With ROS 2 (colcon; the vehicle, scripts/build_deb.sh): the library, the node, the bag replay, the replay tool
#   colcon build --base-paths common_interface node_src/lidar_3d_detection --packages-up-to lidar_3d_detection \
#       --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87 -DLIDAR3D_MODEL_DIR=<model with engines>
#   LIDAR3D_MODEL_DIR installs the model (model_meta.json, the ONNX graphs and the LIDAR3D_PRECISION engines built on
#   this machine) to <prefix>/model, where config/lidar_3d_detection.yaml's node.model_dir points.
#
# Without ROS (plain CMake; the server's Docker image, any machine): the library, the replay tool and the tests
#   cmake -S <this directory> -B build/lidar3d -DLIDAR3D_ROS=OFF -DLIDAR3D_TESTS=ON [-DLIDAR3D_WITH_CUDA=ON]
#   cmake --build build/lidar3d -j && ctest --test-dir build/lidar3d --output-on-failure
#   Inside the CenterPoint repository CTest first writes the golden data (every stage in Python on the same inputs:
#   tools/deploy/dump_golden.py, python3 with numpy and scipy) into the build; elsewhere pass LIDAR3D_GOLDEN_DIR.
#   The GPU tests read the model from the environment variable LIDAR3D_MODEL_DIR.
cmake_minimum_required(VERSION 3.16)
project(lidar_3d_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# no -ffast-math, and a * b + c as two roundings (no FMA contraction, which the Orin's aarch64 GCC
# does by default): the float32 stages must round like NumPy, or points move across voxel edges
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter> $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-dangling-reference HAVE_NO_DANGLING_REFERENCE)
if(HAVE_NO_DANGLING_REFERENCE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>)
endif()

find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  set(_lidar3d_ros_default ON)
else()
  set(_lidar3d_ros_default OFF)
endif()
option(LIDAR3D_ROS "the ROS 2 node and bag replay (colcon); OFF: the library, the replay tool and the tests"
       ${_lidar3d_ros_default})
if(LIDAR3D_ROS)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rosbag2_cpp REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(perception_interface REQUIRED)
  find_package(glog REQUIRED)
  option(LIDAR3D_KYGI "kygi665_msgs/INSPVAX localization (the kygi665 robot)" ON)
  if(LIDAR3D_KYGI)
    find_package(kygi665_msgs REQUIRED)
  endif()
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

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
option(LIDAR3D_TESTS "GoogleTest parity tests against the Python reference" OFF)
set(LIDAR3D_GOLDEN_DIR "" CACHE PATH
    "golden data (tools/deploy/dump_golden.py); empty: CTest writes it into the build (inside the CenterPoint repo)")
# the CenterPoint repository, when this node is its Nav_D20/perception_combo_ros/node_src/lidar_3d_detection
get_filename_component(LIDAR3D_ALGO_REPO ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)
set(LIDAR3D_MODEL_DIR "" CACHE PATH "model to install: model_meta.json, *.onnx and the engines built on this machine")
set(LIDAR3D_PRECISION "fp16" CACHE STRING "engines to install from LIDAR3D_MODEL_DIR (<graph>.<precision>.engine)")

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

add_executable(lidar_3d_replay tools/lidar_3d_replay.cpp)
target_link_libraries(lidar_3d_replay PRIVATE lidar3d)

if(LIDAR3D_ROS)
  set(ROS_DEPS rclcpp rosbag2_cpp std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs perception_interface)
  if(LIDAR3D_KYGI)
    list(APPEND ROS_DEPS kygi665_msgs)
  endif()

  add_library(lidar3d_ros STATIC src/ros_convert.cpp)
  target_include_directories(lidar3d_ros PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(lidar3d_ros PUBLIC lidar3d)
  ament_target_dependencies(lidar3d_ros PUBLIC ${ROS_DEPS})
  if(LIDAR3D_KYGI)
    target_compile_definitions(lidar3d_ros PUBLIC LIDAR3D_HAVE_KYGI=1)
  endif()

  add_executable(lidar_3d_detection src/main.cpp src/lidar_3d_detection.cpp)
  target_include_directories(lidar_3d_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(lidar_3d_detection lidar3d_ros glog::glog)
  ament_target_dependencies(lidar_3d_detection ${ROS_DEPS} ament_index_cpp tf2_ros)

  add_executable(lidar_3d_bag_replay src/bag_replay.cpp)
  target_include_directories(lidar_3d_bag_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(lidar_3d_bag_replay lidar3d_ros)
  ament_target_dependencies(lidar_3d_bag_replay ${ROS_DEPS})
endif()

if(LIDAR3D_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  set(_golden ${LIDAR3D_GOLDEN_DIR})
  if(NOT _golden)
    if(NOT EXISTS ${LIDAR3D_ALGO_REPO}/tools/deploy/dump_golden.py)
      message(FATAL_ERROR "LIDAR3D_TESTS: set LIDAR3D_GOLDEN_DIR (no tools/deploy/dump_golden.py in ${LIDAR3D_ALGO_REPO})")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_golden ${CMAKE_CURRENT_BINARY_DIR}/golden)
    # the Python reference of every stage, recomputed whenever the tests run
    add_test(NAME golden COMMAND ${Python3_EXECUTABLE} ${LIDAR3D_ALGO_REPO}/tools/deploy/dump_golden.py --out ${_golden}
             WORKING_DIRECTORY ${LIDAR3D_ALGO_REPO})
    set_tests_properties(golden PROPERTIES FIXTURES_SETUP golden ENVIRONMENT "PYTHONPATH=${LIDAR3D_ALGO_REPO}")
  endif()
  set(TEST_SOURCES
    tests/test_main.cpp
    tests/test_io_geometry.cpp
    tests/test_tracking.cpp
    tests/test_detection.cpp
    tests/test_preprocess.cpp
    tests/test_pipeline.cpp
    tests/test_online.cpp
  )
  if(LIDAR3D_WITH_CUDA)
    list(APPEND TEST_SOURCES tests/test_gpu.cpp)
  endif()
  add_executable(lidar3d_tests ${TEST_SOURCES})
  target_link_libraries(lidar3d_tests PRIVATE lidar3d GTest::gtest)
  target_include_directories(lidar3d_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(lidar3d_tests PRIVATE LIDAR3D_GOLDEN_DIR="${_golden}"
                             LIDAR3D_REPLAY_BIN="$<TARGET_FILE:lidar_3d_replay>")
  add_test(NAME parity COMMAND lidar3d_tests)
  if(NOT LIDAR3D_GOLDEN_DIR)
    set_tests_properties(parity PROPERTIES FIXTURES_REQUIRED golden)
  endif()
endif()

if(NOT LIDAR3D_ROS)
  return()
endif()

install(TARGETS lidar_3d_detection lidar_3d_bag_replay lidar_3d_replay
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION .)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS packaging/start_lidar_3d_detection.sh packaging/lock_clocks.sh DESTINATION bin)
# optional Fast DDS settings for the large clouds (README "DDS"; scripts/orin_foxglove.sh uses it)
install(FILES packaging/fastdds_large_buffers.xml DESTINATION share/${PROJECT_NAME})

if(LIDAR3D_MODEL_DIR)
  set(_model_files model_meta.json pfn.onnx rpn_head.onnx pfn.${LIDAR3D_PRECISION}.engine
      rpn_head.${LIDAR3D_PRECISION}.engine)
  foreach(f ${_model_files})
    if(NOT EXISTS ${LIDAR3D_MODEL_DIR}/${f})
      message(FATAL_ERROR "LIDAR3D_MODEL_DIR ${LIDAR3D_MODEL_DIR} has no ${f} (engines: build_engines.py on this machine)")
    endif()
    list(APPEND _model_paths ${LIDAR3D_MODEL_DIR}/${f})
  endforeach()
  # ONNX weights stored next to the graph (torch's dynamo exporter), and the record of how and where the
  # engines were built, when there are
  file(GLOB _onnx_data ${LIDAR3D_MODEL_DIR}/*.onnx.data)
  list(APPEND _model_paths ${_onnx_data})
  if(EXISTS ${LIDAR3D_MODEL_DIR}/engines.json)
    list(APPEND _model_paths ${LIDAR3D_MODEL_DIR}/engines.json)
  endif()
  install(FILES ${_model_paths} DESTINATION model)
endif()

ament_package()
