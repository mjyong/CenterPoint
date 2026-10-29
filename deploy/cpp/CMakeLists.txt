# libdogperc: the online perception pipeline of dog_perception in C++ (docs/DEPLOY_ORIN.md, section 7).
#
#   cmake -S deploy/cpp -B deploy/cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build deploy/cpp/build -j && ctest --test-dir deploy/cpp/build --output-on-failure
#
# DOGPERC_WITH_CUDA builds the GPU pillar preprocessing and the TensorRT engines (the server's
# nvcr.io/nvidia/tensorrt:24.08-py3 container, the Orin). Without it the library has the CPU reference
# of every stage and runs the detector from saved head maps or detections only.
# The parity tests compare every stage with the Python implementation: CTest first runs
# tools/deploy/dump_golden.py (python3 with numpy and scipy) into <build>/golden.
cmake_minimum_required(VERSION 3.18)
project(dogperc LANGUAGES CXX)

option(DOGPERC_WITH_CUDA "GPU pillar preprocessing and TensorRT engines" OFF)
option(DOGPERC_TESTS "GoogleTest parity tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# no -ffast-math: the float32 stages must round like NumPy for the voxel parity
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
include(CheckCXXCompilerFlag)
# GCC 13 warns on every reference returned from a function that also takes a temporary (member(npz, "x"))
check_cxx_compiler_flag(-Wno-dangling-reference HAVE_NO_DANGLING_REFERENCE)
if(HAVE_NO_DANGLING_REFERENCE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>)
endif()
# keep a * b + c as two roundings, as NumPy's elementwise float32 code does; GCC contracts to FMA by default
# where FMA is part of the base ISA (aarch64, the Orin), which moves points across voxel edges
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(DOGPERC_SOURCES
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
)

if(DOGPERC_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  # the kernels use round-to-nearest intrinsics anyway; no contraction anywhere else either
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Orin (sm_87) and the server's RTX 4090 (sm_89)
    set(CMAKE_CUDA_ARCHITECTURES 87 89)
  endif()
  find_path(TENSORRT_INCLUDE NvInfer.h HINTS $ENV{TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu
            /usr/include/aarch64-linux-gnu /usr/local/tensorrt/include)
  find_library(NVINFER nvinfer HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  if(NOT TENSORRT_INCLUDE OR NOT NVINFER)
    message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer); set TENSORRT_ROOT")
  endif()
  list(APPEND DOGPERC_SOURCES src/cuda/pillars.cu src/cuda/trt_engine.cpp src/cuda/gpu_detector.cpp)
endif()

add_library(dogperc ${DOGPERC_SOURCES})
target_include_directories(dogperc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(dogperc PUBLIC Eigen3::Eigen yaml-cpp ZLIB::ZLIB Threads::Threads)
if(DOGPERC_WITH_CUDA)
  target_compile_definitions(dogperc PUBLIC DOGPERC_WITH_CUDA=1)
  target_include_directories(dogperc PUBLIC ${TENSORRT_INCLUDE})
  target_link_libraries(dogperc PUBLIC CUDA::cudart ${NVINFER})
  set_target_properties(dogperc PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

add_executable(dogperc_replay tools/dogperc_replay.cpp)
target_link_libraries(dogperc_replay PRIVATE dogperc)

if(DOGPERC_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  find_package(Python3 COMPONENTS Interpreter)
  get_filename_component(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
  set(GOLDEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/golden)
  set(TEST_SOURCES
    tests/test_main.cpp
    tests/test_io_geometry.cpp
    tests/test_tracking.cpp
    tests/test_detection.cpp
    tests/test_preprocess.cpp
    tests/test_pipeline.cpp
  )
  if(DOGPERC_WITH_CUDA)
    list(APPEND TEST_SOURCES tests/test_gpu.cpp)
  endif()
  add_executable(dogperc_tests ${TEST_SOURCES})
  target_link_libraries(dogperc_tests PRIVATE dogperc GTest::gtest)
  target_include_directories(dogperc_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(dogperc_tests PRIVATE DOGPERC_GOLDEN_DIR="${GOLDEN_DIR}"
                             DOGPERC_REPLAY_BIN="$<TARGET_FILE:dogperc_replay>")
  if(Python3_Interpreter_FOUND)
    # the Python reference of every stage, recomputed whenever the tests run
    add_test(NAME golden COMMAND ${Python3_EXECUTABLE} ${REPO_ROOT}/tools/deploy/dump_golden.py --out ${GOLDEN_DIR}
             WORKING_DIRECTORY ${REPO_ROOT})
    set_tests_properties(golden PROPERTIES FIXTURES_SETUP golden ENVIRONMENT "PYTHONPATH=${REPO_ROOT}")
  endif()
  add_test(NAME parity COMMAND dogperc_tests)
  set_tests_properties(parity PROPERTIES FIXTURES_REQUIRED golden)
endif()
