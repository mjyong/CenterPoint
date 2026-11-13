# ============================================================
# Generic Linux Cross Toolchain (Industrial Version)
# Supports: aarch64 / arm / riscv / any gcc-based cross toolchain
# ============================================================

cmake_minimum_required(VERSION 3.16)

# ---------- 检查环境变量 ----------
foreach(var TARGET_ARCH CROSS_COMPILE SYSROOT ROS2_INSTALL_PATH PYTHON_SOABI)
    if(NOT DEFINED ENV{${var}})
        message(FATAL_ERROR "Environment variable ${var} not defined")
    endif()
endforeach()

# ---------- 基础信息 ----------
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR $ENV{TARGET_ARCH})

# ---------- sysroot ----------
if(DETECTED_SYSROOT)
    set(CMAKE_SYSROOT ${DETECTED_SYSROOT})
else()
    set(CMAKE_SYSROOT $ENV{SYSROOT})
endif()
message(STATUS "Using sysroot: ${CMAKE_SYSROOT}  --- ${CROSS_COMPILE}")

# ---------- 交叉编译器 ----------
set(CROSS_COMPILE $ENV{CROSS_COMPILE})
set(CMAKE_C_COMPILER ${CROSS_COMPILE}gcc${COMPILES_SUFFIX})
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++${COMPILES_SUFFIX})
set(CMAKE_ASM_COMPILER ${CROSS_COMPILE}as)
set(CMAKE_LINKER ${CROSS_COMPILE}ld)

# find_program(CMAKE_ASM_COMPILER NAMES ${CROSS_COMPILE}as REQUIRED)
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "ASM Compiler: ${CMAKE_ASM_COMPILER}")
message(STATUS "Linker: ${CMAKE_LINKER}")

# ---------- 编译标志 ----------
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --sysroot=${CMAKE_SYSROOT} -isystem ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=${CMAKE_SYSROOT} -isystem ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")

link_directories(${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
link_directories(${CMAKE_SYSROOT}/usr/lib)

# ============================================================
# 检查 private.cmake
# ============================================================
set(PRIVATE_LINUX_CMAKE ${ROOT_PATH_DIR}/projects/private_linux.cmake)
if(EXISTS ${PRIVATE_LINUX_CMAKE})
  message(STATUS "Using private configuration: ${PRIVATE_LINUX_CMAKE}")
  include(${PRIVATE_LINUX_CMAKE})
else()
  message(STATUS "No private configuration found, using default sources, ${ROOT_PATH_DIR}/projects/private_linux.cmake")
endif()


set(CMAKE_LIBRARY_PATH "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/lib")
set(CMAKE_INCLUDE_PATH "${CMAKE_SYSROOT}/usr/include" "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")

# ---------------------------
# Linker flags
# ---------------------------
set(EXE_LINK_FLAGS "-L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -L${CMAKE_SYSROOT}/usr/lib \
  -Wl,-rpath,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu \
  -B${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXE_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXE_LINK_FLAGS}")

# ---------------------------
# PKG_CONFIG
# ---------------------------
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
set(ENV{PKG_CONFIG_PATH} "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")
# --------------------------- 
# Prefix path 
# --------------------------- 
set(CMAKE_PREFIX_PATH 
    "${CMAKE_SYSROOT}/usr" 
    "${CMAKE_SYSROOT}/usr/lib"
    "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu" 
    "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/cmake" 
    "${CMAKE_PREFIX_PATH}" 
)

set(CMAKE_FIND_ROOT_PATH $ENV{ROS2_INSTALL_PATH} $ENV{SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64-static)

# ---------------------------
# Python 配置
# ---------------------------
set(PYTHON_SOABI "$ENV{PYTHON_SOABI}")
set(PYTHON_MODULE_EXTENSION ".${PYTHON_SOABI}.so")
set(PYTHON_INCLUDE_DIR ${CMAKE_SYSROOT}/usr/include/python3.10)
set(PYTHON_LIBRARY ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/libpython3.10.so)

# Host python for code generation
set(Python3_EXECUTABLE /usr/bin/python3)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------- Threads ----------
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(Threads_FOUND TRUE)

# ---------- Qt5 moc (host tool) ----------
if(NOT TARGET Qt5::moc)
  set(QT_MOC_EXECUTABLE /usr/bin/moc)
  add_executable(Qt5::moc IMPORTED)
  set_property(TARGET Qt5::moc PROPERTY IMPORTED_LOCATION ${QT_MOC_EXECUTABLE})
endif()

# ---------- Debug Info ----------
message(STATUS "--------------------------------------------------")
message(STATUS "Target architecture : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Sysroot             : ${CMAKE_SYSROOT}")
message(STATUS "C Compiler          : ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler        : ${CMAKE_CXX_COMPILER}")
message(STATUS "Linker              : ${CMAKE_LINKER}")
message(STATUS "Assembler           : ${CMAKE_ASM_COMPILER}")
message(STATUS "ROS2 Path           : $ENV{ROS2_INSTALL_PATH}")
message(STATUS "Python SOABI        : ${PYTHON_SOABI}")
message(STATUS "--------------------------------------------------")
