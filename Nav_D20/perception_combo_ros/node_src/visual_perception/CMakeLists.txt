cmake_minimum_required(VERSION 3.10)

# 在文件开头附近添加CMAKE配置
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加RPATH相关设置
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

PROJECT(visual_perception VERSION 1.0.0 LANGUAGES C CXX)

message(WARNING "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}" )
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64" )
message(WARNING "xxxxxxxxxxxxxxxxxxxxxx" )
  set(ON_ORIN ON)
  message(WARNING "ON_ORIN: ${ON_ORIN}" )
else()
  set(ON_ORIN OFF)
  message(WARNING "ON_ORIN: ${ON_ORIN}" )
endif()

set(ON_LAPTOP ON)
# set(ON_ORIN_DEBUG ON) # 手动在orin上编译测试

# set ON_ORIN_DEBUG =${ON_ORIN_DEBUG}
message(WARNING "ON_ORIN_DEBUG: ${ON_ORIN_DEBUG}" )

option(ENABLE_ROS2 "Usr Ros2." ON)
message(STATUS "ENABLE_ROS2: ${ENABLE_ROS2}" )
option(ENABLE_SERES_IPC "Usr Seres ipc." OFF)
message(STATUS "ENABLE_SERES_IPC: ${ENABLE_SERES_IPC}" )

# eigen库和opencv库
find_package(Eigen3 REQUIRED) 
find_package(OpenCV REQUIRED)
message(STATUS "Detected OPENCV version: ${OPENCV_VERSION}")

# trt和cuda库
message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
set( CMAKE_CXX_FLAGS "-O3" )                 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

if(${ON_ORIN_DEBUG})
  find_package(CUDA REQUIRED)
  find_library(CUDA_RUNTIME_LIB cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  list(APPEND ALL_LIBS 
  #   ${CUDA_LIBRARIES} 
    ${CUDA_RUNTIME_LIB}
    ${CUDA_cublas_LIBRARY} 
    ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})
  # list(APPEND ALL_LIBS 
  #     CUDA::cudart
  #     CUDA::cublas
  #     CUDA::nppc
  #     CUDA::nppig
  #     CUDA::nppial
  # )

  list(APPEND INCLUDE_DRIS ${CUDA_INCLUDE_DIRS})
  set(TensorRT_LIB_ROOT /usr/lib/aarch64-linux-gnu)
  set(TensorRT_INCLUDE_ROOT /usr/include/aarch64-linux-gnu)
  find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_LIB_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_LIB_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_LIB_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  # find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_INCLUDE_ROOT} PATH_SUFFIXES include)
  list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER} 
      #  ${TRT_NVCAFFE_PARSER}
      )

  # include tensorrt's headers
  list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})
  set(SAMPLES_COMMON_DIR ${TensorRT_LIB_ROOT}/samples/common)
  list(APPEND INCLUDE_DRIS ${SAMPLES_COMMON_DIR})
  # 手动设置TensorRT包含目录和库目录
  set(TENSORRT_INCLUDE_DIR "${TensorRT_INCLUDE_ROOT}/include")
  set(TENSORRT_LIB_DIR "${TensorRT_LIB_ROOT}/lib")
  message(STATUS TensorRT_LIB_ROOT = ${TensorRT_LIB_ROOT})
  message(STATUS ${ALL_LIBS})

elseif(ON_ORIN)
  find_package(CUDAToolkit REQUIRED)
  find_library(CUDA_RUNTIME_LIB cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  set(TensorRT_LIB_ROOT /l4t/jetson_arm64_sysroot/usr/lib/aarch64-linux-gnu)
  set(TensorRT_INCLUDE_ROOT /l4t/jetson_arm64_sysroot/usr/include/aarch64-linux-gnu)
  
  set(ALL_LIBS
    CUDA::cudart
    CUDA::cublas  
    CUDA::nppc
    CUDA::nppig
    CUDA::nppial
  )

  list(APPEND INCLUDE_DRIS ${CUDA_INCLUDE_DIRS})
  # 设置所有可能的库搜索路径
  set(ALL_JETSON_LIB_PATHS
    /l4t/jetson_arm64_sysroot/usr/lib/aarch64-linux-gnu
    /l4t/jetson_arm64_sysroot/lib/aarch64-linux-gnu
    /l4t/jetson_arm64_sysroot/usr/lib
    /l4t/jetson_arm64_sysroot/lib
    /l4t/jetson_arm64_sysroot/usr/lib/aarch64-linux-gnu/tegra
    /l4t/jetson_arm64_sysroot/usr/local/cuda/targets/aarch64-linux/lib
  )
  
  # 设置rpath链接
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,/l4t/jetson_arm64_sysroot/usr/lib/aarch64-linux-gnu:/l4t/jetson_arm64_sysroot/lib/aarch64-linux-gnu:/l4t/jetson_arm64_sysroot/usr/lib:/l4t/jetson_arm64_sysroot/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,/l4t/jetson_arm64_sysroot/usr/lib/aarch64-linux-gnu:/l4t/jetson_arm64_sysroot/lib/aarch64-linux-gnu:/l4t/jetson_arm64_sysroot/usr/lib:/l4t/jetson_arm64_sysroot/lib")
  
  # 查找所有必需的TensorRT和CUDA库
  find_library(TRT_NVINFER nvinfer PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(TRT_NVINFER_PLUGIN nvinfer_plugin PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(TRT_NVONNX_PARSER nvonnxparser PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(NVDLA_COMPILER nvdla_compiler PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(NVDLA_RUNTIME nvdla_runtime PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(NVRM_GPU nvrm_gpu PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(NVRM_MEM nvrm_mem PATHS ${ALL_JETSON_LIB_PATHS})
  find_library(NVCUDLA nvcudla PATHS ${ALL_JETSON_LIB_PATHS})
  
  # 检查并添加库
  set(REQUIRED_LIBS
    TRT_NVINFER
    TRT_NVINFER_PLUGIN
    TRT_NVONNX_PARSER
    NVDLA_COMPILER
    NVDLA_RUNTIME
    NVRM_GPU
    NVRM_MEM
    NVCUDLA
  )
  
  foreach(LIB ${REQUIRED_LIBS})
    if(${LIB})
      list(APPEND ALL_LIBS ${${LIB}})
      message(STATUS "Found ${LIB}: ${${LIB}}")
    else()
      message(WARNING "Required library ${LIB} not found")
    endif()
  endforeach()
  
  # 包含目录
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h 
            PATHS ${TensorRT_INCLUDE_ROOT}
            PATH_SUFFIXES include)
  list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})

elseif(ON_LAPTOP)

  # trt和cuda库
  message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
  set( CMAKE_CXX_FLAGS "-O3" )                 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  # find thirdparty
  find_package(CUDA REQUIRED)
  find_library(CUDA_RUNTIME_LIB cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  list(APPEND ALL_LIBS 
  #   ${CUDA_LIBRARIES} 
    ${CUDA_RUNTIME_LIB}
    ${CUDA_cublas_LIBRARY} 
    ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})
  # list(APPEND ALL_LIBS 
  #     CUDA::cudart
  #     CUDA::cublas
  #     CUDA::nppc
  #     CUDA::nppig
  #     CUDA::nppial
  # )

  list(APPEND INCLUDE_DRIS ${CUDA_INCLUDE_DIRS})


  set(TensorRT_ROOT /home/huanghanlin/hhl_work/seres_work/Study/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-11.8/TensorRT-10.1.0.27)
  find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  # find_library(TRT_NVCAFFE_PARSER NAMES nvcaffe_parser HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)
  list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER} 
      #  ${TRT_NVCAFFE_PARSER}
      )

  # include tensorrt's headers
  list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})

  set(SAMPLES_COMMON_DIR ${TensorRT_ROOT}/samples/common)
  list(APPEND INCLUDE_DRIS ${SAMPLES_COMMON_DIR})
  message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
  message(STATUS "ALL_LIBS: ${ALL_LIBS}")


  # 手动设置TensorRT包含目录和库目录
  set(TENSORRT_INCLUDE_DIR "${TensorRT_ROOT}/include")
  set(TENSORRT_LIB_DIR "${TensorRT_ROOT}/lib")
  message(STATUS TensorRT_ROOT = ${TensorRT_ROOT})
  message(STATUS ${ALL_LIBS})


else()
  find_package(CUDAToolkit REQUIRED)

  # 基础CUDA库
  set(ALL_LIBS
      CUDA::cudart
      CUDA::cublas
  )

  # 条件添加NPP库
  if(TARGET CUDA::nppc AND TARGET CUDA::nppig AND TARGET CUDA::nppial)
      list(APPEND ALL_LIBS CUDA::nppc CUDA::nppig CUDA::nppial)
      message(WARNING "Using CUDA NPP libraries")
  else()
      message(WARNING "CUDA NPP targets not available, skipping NPP libraries")
  endif()

  list(APPEND INCLUDE_DRIS ${CUDAToolkit_INCLUDE_DIRS})

  # x86_64的配置保持不变
  set(TensorRT_ROOT /usr/lib/x86_64-linux-gnu)
  set(TensorRT_INCLUDE_ROOT /usr/include/x86_64-linux-gnu)
  
  find_library(TRT_NVINFER NAMES nvinfer HINTS ${TensorRT_ROOT})
  find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin HINTS ${TensorRT_ROOT})
  find_library(TRT_NVONNX_PARSER NAMES nvonnxparser HINTS ${TensorRT_ROOT})
  
  list(APPEND ALL_LIBS ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN} ${TRT_NVONNX_PARSER})
  list(APPEND INCLUDE_DRIS ${TENSORRT_INCLUDE_DIR})
endif()



message(STATUS ***INCLUDE_DRIS*** = ${INCLUDE_DRIS})
message(STATUS "ALL_LIBS: ${ALL_LIBS}")


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
    
    # 尝试查找nav_interface，但不强制要求
    find_package(nav_interface REQUIRED)
    # find_package(ament_cmake REQUIRED)
    
    # glog 日志库
    find_package(glog REQUIRED)
    

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
        ${std_msgs_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${geometry_msgs_INCLUDE_DIRS}
        ${visualization_msgs_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${nav_interface_INCLUDE_DIRS}
    )
    
    set(ROS_LIBRARIES
        ${rclcpp_LIBRARIES}
        ${std_msgs_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${geometry_msgs_LIBRARIES}
        ${visualization_msgs_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${nav_interface_LIBRARIES}
        glog::glog
    )
    
    # if(nav_interface_FOUND)
    #     list(APPEND ROS_INCLUDE_DIRS ${nav_interface_INCLUDE_DIRS})
    #     list(APPEND ROS_LIBRARIES ${nav_interface_LIBRARIES})
    #     message(STATUS "nav_interface found and added")
    # else()
    #     message(STATUS "nav_interface not found, skipping")
    # endif()
    
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
#   ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_seg/utils/*.cu
#   ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_seg/utils/*.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_seg/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_seg/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/yolov8_detect/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(REMOVE_ITEM CPPS src/main.cpp)
message(STATUS CPPS = ${CPPS})
list (LENGTH CPPS length)
message(STATUS ***length*** = ${length}) 


include_directories(${CUR_DIR}/include
                    ${CUR_DIR}/yolov8_seg
                    ${INCLUDE_DRIS}
                    ${OPENCV_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
                    ${ROS_INCLUDE_DIRS}
                    ${IPC_INCLUDE_DIRS}
                    )


add_library(${PROJECT_NAME} SHARED ${CPPS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUR_DIR}/include
    ${CUR_DIR}/yolov8_seg
    ${INCLUDE_DRIS}
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
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES  60 61 62 70 72 75 86)
target_compile_options(${PROJECT_NAME} PUBLIC 
   $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo; --use_fast_math --disable-warnings>)

add_executable(visual_perception_2d
                src/main.cpp)

set(VISUAL_PERCEPTION_DEPENDENCIES
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  nav_interface
  glog
)

ament_target_dependencies(visual_perception_2d ${VISUAL_PERCEPTION_DEPENDENCIES}) 

# 设置目标属性来解决循环依赖
set_target_properties(visual_perception_2d PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
)

target_link_libraries(visual_perception_2d 
                      ${PROJECT_NAME} 
                      ${ALL_LIBS} 
                      ${OpenCV_LIBRARIES} 
                      ${ROS_LIBRARIES} 
                      ${SERES_IPC_LIBRARIES} 
                      glog::glog
                      )

install(TARGETS
${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
    visual_perception_2d
    DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY config DESTINATION ./)
install(DIRECTORY model DESTINATION ./)

# ament_package()
