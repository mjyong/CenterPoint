#!/bin/bash
# 交叉编译版本：在 x86_64 Docker 中为 aarch64 (Jetson Orin) 编译感知模块
# 参照 script/script/build_control.sh 与 script/toolchain_aarch64.cmake 中的交叉编译规范
#
# usage:
#   ./build_perception_crosscompile.sh                    # 正常交叉编译
#   ./build_perception_crosscompile.sh -d                 # 清理所有构建文件
#   ./build_perception_crosscompile.sh -s <sysroot路径>   # 指定 sysroot 路径
#   ./build_perception_crosscompile.sh -c <toolchain路径> # 指定 toolchain.cmake 路径
#   ./build_perception_crosscompile.sh -b <Debug|Release|RelWithDebInfo>

# ============================================================
# 日志函数（与 script/script/import_log_func 保持一致）
# ============================================================
function log_error() { echo -e "\E[1;31m$(date '+%Y-%m-%d %H:%M:%S') [Error] $*\033[0m"; }
function log_ok()    { echo -e "\E[1;32m$(date '+%Y-%m-%d %H:%M:%S') [ OK ] $*\033[0m"; }
function log_warn()  { echo -e "\E[33m$(date '+%Y-%m-%d %H:%M:%S') [Warn] $*\033[0m"; }
function log_info()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [Info] $*"; }

# ============================================================
# 0. 基础路径配置
# ============================================================
script_abs_file_path=$(realpath "$0")
script_abs_dir_path=$(dirname ${script_abs_file_path})
export SERES_PERCEPTION_ROOT_DIR=${script_abs_dir_path}
export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install
export ON_ORIN_DEBUG=OFF

log_info "perception crosscompile script: ${script_abs_file_path}"
log_info "SERES_PERCEPTION_ROOT_DIR: ${SERES_PERCEPTION_ROOT_DIR}"

# ============================================================
# 交叉编译默认配置（可通过命令行参数覆盖）
# toolchain 文件位于上层 script/ 目录
# ============================================================
SYSROOT_DIR="/l4t/jetson_arm64_sysroot"
TOOLCHAIN_FILE="${SERES_PERCEPTION_ROOT_DIR}/toolchain_aarch64.cmake"
build_type="Release"

# ============================================================
# 清理函数
# ============================================================
clean_build() {
    log_warn "开始清理构建文件..."
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/install
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/node_src/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/node_src/log
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/log
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/common_interface/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/common_interface/log
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/glog-0.6.0/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/gflags/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/log
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/octomap/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/pcl_msgs/build
    rm -rf ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/octomap_msgs/build
    log_ok "所有构建文件清理完成!"
    exit 0
}

# ============================================================
# 处理命令行参数
# ============================================================
while getopts "ds:c:b:" opt; do
    case $opt in
        d) clean_build ;;
        s) SYSROOT_DIR="${OPTARG}" ;;
        c) TOOLCHAIN_FILE="${OPTARG}" ;;
        b) build_type="${OPTARG}" ;;
        *)
            echo "usage: $0 [-d] [-s <sysroot路径>] [-c <toolchain.cmake路径>] [-b <Debug|Release|RelWithDebInfo>]"
            exit 1
            ;;
    esac
done

# ============================================================
# 校验参数
# ============================================================
TOOLCHAIN_FILE=$(realpath "${TOOLCHAIN_FILE}" 2>/dev/null || echo "${TOOLCHAIN_FILE}")
if [ ! -f "${TOOLCHAIN_FILE}" ]; then
    log_error "toolchain 文件不存在: ${TOOLCHAIN_FILE}"
    log_error "请通过 -c 参数指定路径，例如: -c /path/to/toolchain_aarch64.cmake"
    exit 1
fi
if [ ! -d "${SYSROOT_DIR}" ]; then
    log_error "sysroot 目录不存在: ${SYSROOT_DIR}"
    log_error "请通过 -s 参数指定路径，例如: -s /l4t/jetson_arm64_sysroot"
    exit 1
fi
if [ "${build_type}" != "Debug" ] && [ "${build_type}" != "Release" ] && [ "${build_type}" != "RelWithDebInfo" ]; then
    log_error "无效的构建类型: ${build_type}，可选: Debug Release RelWithDebInfo"
    exit 1
fi

# ============================================================
# 初始化交叉编译环境（与 build_control.sh 的 init_build_env 保持一致）
# ============================================================
init_cross_compile_env() {
    log_info "初始化交叉编译环境..."

    # --- 编译器配置 ---
    export TARGET_ARCH="aarch64"
    export CROSS_COMPILE=/usr/bin/aarch64-linux-gnu-
    export CC=${CROSS_COMPILE}gcc
    export CXX=${CROSS_COMPILE}g++
    export LD=${CROSS_COMPILE}ld
    export RANLIB=${CROSS_COMPILE}ranlib
    export STRIP=${CROSS_COMPILE}strip

    # --- sysroot（toolchain.cmake 通过 $ENV{SYSROOT} 读取，必须 export）---
    export SYSROOT=${SYSROOT_DIR}

    # --- ROS 路径 ---
    HOST_ROS_PATH="/opt/ros/humble"
    TARGET_ROS_PATH="${SYSROOT}/opt/ros/humble"
    PYTHON_VERSION="python3.10"

    # 清理宿主机 ROS 环境，避免污染交叉编译
    unset ROS_DISTRO
    unset CMAKE_PREFIX_PATH
    unset AMENT_PREFIX_PATH

    # source 宿主机 ROS，仅使 colcon/ament 工具链自身可用
    if [ -f "${HOST_ROS_PATH}/setup.bash" ]; then
        log_info "Sourcing host ROS2 environment: ${HOST_ROS_PATH}/setup.bash"
        source "${HOST_ROS_PATH}/setup.bash"
    else
        log_error "宿主机 ROS2 未找到: ${HOST_ROS_PATH}/setup.bash"
        exit 1
    fi

    # --- toolchain.cmake 的 CMAKE_FIND_ROOT_PATH 中引用的变量，必须全部 export ---
    # ROS2_INSTALL_PATH: 感知节点最终安装目录，作为 CMake find 查找路径之一
    export ROS2_INSTALL_PATH=${SERES_PERCEPTION_INSTALL_DIR}
    # SERES_INTERFACE_INSTALL_DIR: 接口消息包安装目录（perception 复用同一 install 目录）
    export SERES_INTERFACE_INSTALL_DIR=${SERES_PERCEPTION_INSTALL_DIR}
    # MID_INSTALL_DIR: 中间件安装目录（perception 无独立中间件，复用 install 目录）
    export MID_INSTALL_DIR=${SERES_PERCEPTION_INSTALL_DIR}
    # SYSROOT_CUDA_HOME: toolchain.cmake CMAKE_FIND_ROOT_PATH 首项
    export SYSROOT_CUDA_HOME="${SYSROOT}/usr/local/cuda"
    export PYTHON_SOABI=cpython-310-aarch64-linux-gnu

    # --- 交叉编译时 ROS 相关环境变量 ---
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_DISTRO=humble
    export ROS_LOCALHOST_ONLY=0
    export ROS_PACKAGE_PATH=${TARGET_ROS_PATH}/share
    # CMAKE_PREFIX_PATH 指向 sysroot 内的 ROS 及已编译产物
    export CMAKE_PREFIX_PATH="${TARGET_ROS_PATH}:${SERES_PERCEPTION_INSTALL_DIR}"
    export PKG_CONFIG_PATH="${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig"
    export PKG_CONFIG_SYSROOT_DIR="${SYSROOT}"
    export PYTHON_EXECUTABLE_PATH="/usr/bin/python3"
    export PYTHONPATH="${HOST_ROS_PATH}/local/lib/${PYTHON_VERSION}/dist-packages:${HOST_ROS_PATH}/lib/${PYTHON_VERSION}/site-packages"

    # --- CUDA 配置（指向 sysroot 内的 CUDA，与 build_control.sh 一致）---
    CUDA_HOME="/usr/local/cuda"
    CUDA_HOME_ARM64="${SYSROOT}${CUDA_HOME}"
    export CUDA_HOME=${CUDA_HOME_ARM64}
    export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME_ARM64}/bin   # toolchain.cmake 通过 $ENV{CUDA_TOOLKIT_ROOT_DIR} 读取
    export CUDA_INCLUDE_DIRS=${CUDA_HOME_ARM64}/include
    export CUDA_LIBRARY_DIRS=${CUDA_HOME_ARM64}/lib64
    export CUDA_CUDART_LIBRARY=${CUDA_HOME_ARM64}/lib64/libcudart.so
    export CUDACXX=${CUDA_HOME_ARM64}/bin/nvcc
    export CUDA_GPU_CC="87"  # Jetson Orin 系列 Compute Capability，toolchain.cmake 通过 $ENV{CUDA_GPU_CC} 读取

    export PATH=${CUDA_HOME_ARM64}/bin:${PATH}
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${HOST_ROS_PATH}/opt/rviz_ogre_vendor/lib:${HOST_ROS_PATH}/lib/x86_64-linux-gnu:${HOST_ROS_PATH}/lib:${LD_LIBRARY_PATH}"

    # --- 头文件 / 链接库搜索路径 ---
    export CPATH="${SYSROOT}/usr/local/include:${TARGET_ROS_PATH}/include:${CUDA_HOME_ARM64}/include:${CPATH}"
    export LIBRARY_PATH="${SYSROOT}/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH}"

    # --- CMake 交叉编译参数（与 build_control.sh 保持一致：只传 toolchain + processor）---
    # toolchain.cmake 自身已处理 SYSROOT / CMAKE_FIND_ROOT_PATH 等，无需重复传入
    CMAKE_CCP_PARAM="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH}"

    log_info "  SYSROOT_DIR          : ${SYSROOT_DIR}"
    log_info "  TOOLCHAIN_FILE       : ${TOOLCHAIN_FILE}"
    log_info "  TARGET_ARCH          : ${TARGET_ARCH}"
    log_info "  build_type           : ${build_type}"
    log_info "  CUDA_HOME (arm64)    : ${CUDA_HOME_ARM64}"
    log_info "  CMAKE_PREFIX_PATH    : ${CMAKE_PREFIX_PATH}"
    log_info "  SERES_PERCEPTION_INSTALL_DIR: ${SERES_PERCEPTION_INSTALL_DIR}"
}

init_cross_compile_env

# ============================================================
# 待编译的感知节点
# ============================================================
colcon_compile_node=(
    visual_perception
    pedestrian_code_conduct
    lidar_seg
    grid_map_memory_node
    ptz_data_detection
    perception_recharge
)

start_build_time=$(date +%Y-%m-%d-%H-%M-%S)
log_ok "Starts build time: ${start_build_time}"

# ============================================================
# 1. 交叉编译 gflags
# ============================================================
log_info "============================ 开始交叉编译 gflags ============================"
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/gflags
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/gflags \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    ${CMAKE_CCP_PARAM}
make -j$(nproc) && make install
if [ $? -ne 0 ]; then log_error "gflags 交叉编译失败"; exit 1; fi
log_ok "============================ compile gflags done !============================"

# ============================================================
# 2. 交叉编译 glog
# ============================================================
log_info "============================ 开始交叉编译 glog ============================"
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/glog-0.6.0
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/glog-0.6.0 \
    -DWITH_GFLAGS=ON \
    -DCMAKE_PREFIX_PATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags \
    -DCMAKE_INSTALL_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags/lib \
    -DCMAKE_BUILD_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags/lib \
    -DWITH_UNWIND=OFF \
    ${CMAKE_CCP_PARAM}
make -j$(nproc) && make install
if [ $? -ne 0 ]; then log_error "glog 交叉编译失败"; exit 1; fi
log_ok "============================ compile glog done !============================"

# ============================================================
# 3. 交叉编译 octomap
# ============================================================
log_info "============================ 开始交叉编译 octomap ============================"
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/octomap
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/octomap \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_OCTOVIS_SUBPROJECT=OFF \
    -DCMAKE_INSTALL_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap/lib \
    -DCMAKE_BUILD_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap/lib \
    ${CMAKE_CCP_PARAM}
make -j$(nproc) && make install
if [ $? -ne 0 ]; then log_error "octomap 交叉编译失败"; exit 1; fi
log_ok "============================ compile octomap done !============================"

# ============================================================
# 4. 交叉编译 pcl_msgs 和 octomap_msgs 消息包
# ============================================================
log_info "============================ 开始交叉编译 pcl_msgs & octomap_msgs ============================"
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related
colcon build --paths pcl_msgs octomap_msgs \
    --install-base ${SERES_PERCEPTION_INSTALL_DIR} \
    --allow-overriding octomap_msgs pcl_msgs \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=${build_type} \
        "-DCMAKE_PREFIX_PATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap;${CMAKE_PREFIX_PATH}" \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH}
if [ $? -ne 0 ]; then log_error "pcl_msgs / octomap_msgs 交叉编译失败"; exit 1; fi

source ${SERES_PERCEPTION_INSTALL_DIR}/setup.bash
log_ok "============================ build & source pcl_msgs & octomap_msgs done !============================"

# ============================================================
# 5. 交叉编译 common_interface 消息包
# ============================================================
log_info "============================ 开始交叉编译 common_interface ============================"
cd ${SERES_PERCEPTION_ROOT_DIR}/common_interface/
colcon build \
    --install-base ${SERES_PERCEPTION_INSTALL_DIR} \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH}
if [ $? -ne 0 ]; then log_error "common_interface 交叉编译失败"; exit 1; fi

source ${SERES_PERCEPTION_INSTALL_DIR}/setup.bash
log_ok "============================ build & source common_interface done !============================"

# ============================================================
# 6. 交叉编译感知所有节点
# ============================================================
log_info "============================ 开始交叉编译感知节点 ============================"
export COM_IPC=OFF
cd ${SERES_PERCEPTION_ROOT_DIR}/node_src/
colcon build \
    --paths "./*" \
    --packages-select ${colcon_compile_node[@]} \
    --install-base ${SERES_PERCEPTION_INSTALL_DIR} \
    --cmake-args \
        -DCOM_IPC=${COM_IPC} \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DSERES_PERCEPTION_ROOT_DIR=${SERES_PERCEPTION_ROOT_DIR} \
        -DON_ORIN_DEBUG=${ON_ORIN_DEBUG} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
        -DCUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS} \
        -DCUDA_LIBRARY_DIRS=${CUDA_LIBRARY_DIRS} \
        -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY}
if [ $? -ne 0 ]; then log_error "感知节点交叉编译失败"; exit 1; fi
log_ok "============================ 感知节点交叉编译完成 !============================"

# ============================================================
# 7. 清理产物中的 sysroot 绝对路径（与 build_control.sh 末尾一致）
# ============================================================
log_info "清理 install 目录中 cmake 文件内的 sysroot 绝对路径..."
find ${SERES_PERCEPTION_INSTALL_DIR} -name "*.cmake" -exec sed -i "s|${SYSROOT_DIR}||g" {} \;

log_ok "Build all success"
log_ok "Starts build time: ${start_build_time}"
log_ok "Finish build time: $(date +%Y-%m-%d-%H-%M-%S)"
log_ok "产物目录: ${SERES_PERCEPTION_INSTALL_DIR}"
