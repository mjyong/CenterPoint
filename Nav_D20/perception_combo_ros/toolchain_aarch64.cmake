if("$ENV{TARGET_ARCH}" STREQUAL "" OR
   "$ENV{CROSS_COMPILE}" STREQUAL "" OR
   "$ENV{SYSROOT}" STREQUAL "" OR
   "$ENV{CUDA_GPU_CC}" STREQUAL "" OR
   "$ENV{CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "" OR
   "$ENV{SERES_INTERFACE_INSTALL_DIR}" STREQUAL "" OR
   "$ENV{MID_INSTALL_DIR}" STREQUAL "" OR
   "$ENV{ROS2_INSTALL_PATH}" STREQUAL "")
    message(FATAL_ERROR "Target environment variables not defined")
endif()
#  OR  "$ENV{PYTHON_SOABI}" STREQUAL ""

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR $ENV{TARGET_ARCH})

# Specify the cross compiler
set(CMAKE_C_COMPILER $ENV{CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER $ENV{CROSS_COMPILE}g++)

# Specify the target file system
set(CMAKE_SYSROOT $ENV{SYSROOT})
# set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
# set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE_PATH})
set(PYTHON_SOABI $ENV{PYTHON_SOABI})
set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE_PATH})
# set(PYTHON_PythonInterp /usr/bin/python3.10)
# set(PYTHON_INCLUDE_DIR /usr/include/python3.10)
# set(PYTHON_LIBRARIES /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/libpython3.10.a)
#find_package(PythonInterp REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# set(CMAKE_FIND_ROOT_PATH $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{ROS2_INSTALL_PATH} $ENV{SYSROOT})
set(CMAKE_FIND_ROOT_PATH  $ENV{SYSROOT_CUDA_HOME} $ENV{SYSROOT}/opt/ros/humble  
                          $ENV{ROS2_INSTALL_PATH} $ENV{SERES_INTERFACE_INSTALL_DIR}  
                          $ENV{MID_INSTALL_DIR} $ENV{SYSROOT})

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# 传递sysroot给编译器
# set(CMAKE_C_FLAGS "--sysroot=${CMAKE_SYSROOT} ${CMAKE_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "--sysroot=${CMAKE_SYSROOT} ${CMAKE_CXX_FLAGS}")

#  -DSuiteSparse_CHECK_PATH_SUFFIXES="suitesparse;aarch64-linux-gnu"
# 添加find_library额外的库搜索路径
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${CMAKE_SYSROOT}/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")

# 添加find_package额外的包搜索路径, CMAKE_SYSTEM_PREFIX_PATH为系统自动生成,CMAKE_PREFIX_PATH才是用户添加的
# list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SYSROOT}/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")  # 貌似没必要

set(PROTOBUF_PROTOC_EXECUTABLE /usr/local/bin/protoc)
#####################CUDA 相关配置#################
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_GPU_CC})  # 对应 Jetson 的 GPU 架构（如 Xavier 是 72，Jetson Orin Series 是 87）,查询网址:https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})  # 绑定 AArch64 宿主编译器



###################################################################################
if(NOT TARGET Qt5::moc)
  set(QT_MOC_EXECUTABLE /usr/bin/moc)
  add_executable(Qt5::moc IMPORTED)
  set_property(TARGET Qt5::moc PROPERTY IMPORTED_LOCATION ${QT_MOC_EXECUTABLE})
endif()
if(NOT TARGET Qt5::uic)
  set(QT_UIC_EXECUTABLE /usr/bin/uic)
  add_executable(Qt5::uic IMPORTED)
  set_property(TARGET Qt5::uic PROPERTY IMPORTED_LOCATION ${QT_UIC_EXECUTABLE})
endif()
if(NOT TARGET Qt5::rcc)
  set(QT_RCC_EXECUTABLE /usr/bin/rcc)
  add_executable(Qt5::rcc IMPORTED)
  set_property(TARGET Qt5::rcc PROPERTY IMPORTED_LOCATION ${QT_RCC_EXECUTABLE})
endif()

