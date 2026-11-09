find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
# find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(novatel_oem7_msgs REQUIRED)
find_package(kygi665_msgs REQUIRED)
# GeographicLib: use bundled thirdparty libs instead of find_package
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8|arm")
    set(GEOGRAPHICLIB_ARCH_DIR "ARM")
else()
    set(GEOGRAPHICLIB_ARCH_DIR "X86")
endif()
set(GeographicLib_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/geographiclib/${GEOGRAPHICLIB_ARCH_DIR}/include)
set(GeographicLib_LIBRARIES ${PROJECT_SOURCE_DIR}/thirdparty/geographiclib/${GEOGRAPHICLIB_ARCH_DIR}/lib/libGeographicLib.so)
find_package (livox_ros_driver2 REQUIRED)
find_package (basic REQUIRED)
find_package (nav2_interface REQUIRED)

# OpenMP 并行优化
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    link_libraries(OpenMP::OpenMP_CXX)
endif ()

# 检测处理器架构并设置优化选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386|x86")
    # x86 / x86_64 平台
    if(BUILD_WITH_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        # 启用 AVX2 + FMA 指令集（现代x86 CPU普遍支持，提供更好的向量化性能）
        add_compile_options(-mavx2 -mfma -msse4.2)
    endif()
    # x86特有的优化：启用全寄存器分配优化
    add_compile_options(-fomit-frame-pointer)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    # ARM64 平台（NVIDIA Orin, Apple Silicon, Raspberry Pi 4 等）
    if(BUILD_WITH_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        # NVIDIA Orin 使用 ARMv8.2-A 架构的 Carmel CPU，支持更多扩展
        add_compile_options(-march=armv8.2-a+simd+fp16+crypto+rcpc+dotprod)
    endif()
    # ARM64特有的优化：启用PAC（指针认证）优化栈保护（可选）
    # add_compile_options(-mbranch-protection=standard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    # 32-bit ARM
    if(BUILD_WITH_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        add_compile_options(-mfpu=neon -mfloat-abi=hard -march=armv7-a)
    endif()
else()
    # 其他架构（RISC-V 等）：不加 SIMD 选项
    if(BUILD_WITH_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# 启用链接时优化（LTO）- 跨模块优化，提升整体性能
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_result OUTPUT ipo_output)
if(ipo_result)
    message(STATUS "LTO / IPO optimization is supported, enabling...")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(STATUS "LTO / IPO optimization not supported: ${ipo_output}")
endif()

# 启用循环向量化优化提示
add_compile_options(-ftree-vectorize -fvect-cost-model=dynamic)

# 启用函数内联优化
add_compile_options(-finline-functions -finline-limit=65535)

# 启用优化跳转表
add_compile_options(-foptimize-sibling-calls)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        # ${Pangolin_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${tf2_INCLUDE_DIRS}
        ${pcl_conversions_INCLUDR_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${rosbag2_cpp_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
                ${GeographicLib_INCLUDE_DIR}
		/usr/include/opencv4/
)




include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty
)


set(third_party_libs
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        # ${Pangolin_LIBRARIES}
        glog gflags
        ${yaml-cpp_LIBRARIES}
        ${pcl_conversions_LIBRARIES}
        tbb
        ${rosbag2_cpp_LIBRARIES}
                ${GeographicLib_LIBRARIES}
)

