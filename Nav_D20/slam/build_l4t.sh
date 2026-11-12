#!/bin/bash

work_dir="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$work_dir"
project_name=$(basename $work_dir)
temp_name=.temp
package_dir=$work_dir/$temp_name/$project_name
cpu_num=`cat /proc/stat |grep cpu[0-9] -c`
cpu_num=$[$cpu_num-4]
echo "cpu_num: $cpu_num"
ssh_options="-o StrictHostKeyChecking=no"
remote_user="nvidia"
remote_pwd="nvidia"

# ============================================================
# 日志函数
# ============================================================
function log_error() { echo -e "\E[1;31m$(date '+%Y-%m-%d %H:%M:%S') [Error] $*\033[0m"; }
function log_ok()    { echo -e "\E[1;32m$(date '+%Y-%m-%d %H:%M:%S') [ OK ] $*\033[0m"; }
function log_warn()  { echo -e "\E[33m$(date '+%Y-%m-%d %H:%M:%S') [Warn] $*\033[0m"; }
function log_info()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [Info] $*"; }

# ============================================================
# 交叉编译默认配置
# ============================================================
SYSROOT_DIR="/l4t/jetson_arm64_sysroot"
TOOLCHAIN_FILE="${work_dir}/toolchain-aarch64.cmake"
build_type="Release"

function init_cross_compile_env() {
    log_info "初始化交叉编译环境..."

    # --- 编译器配置 ---
    export TARGET_ARCH="aarch64"
    export CROSS_COMPILE=/usr/bin/aarch64-linux-gnu-
    export CC=${CROSS_COMPILE}gcc
    export CXX=${CROSS_COMPILE}g++
    export LD=${CROSS_COMPILE}ld
    export RANLIB=${CROSS_COMPILE}ranlib
    export STRIP=${CROSS_COMPILE}strip

    # 强制使用宿主机的 make
    export MAKE=/usr/bin/make
    
    # --- sysroot ---
    export SYSROOT=${SYSROOT_DIR}

    # --- ROS 路径 ---
    HOST_ROS_PATH="/opt/ros/humble"
    TARGET_ROS_PATH="${SYSROOT}/opt/ros/humble"
    PYTHON_VERSION="python3.10"

    # 清理宿主机 ROS 环境
    unset ROS_DISTRO
    unset CMAKE_PREFIX_PATH
    unset AMENT_PREFIX_PATH

    # source 宿主机 ROS
    if [ -f "${HOST_ROS_PATH}/setup.bash" ]; then
        log_info "Sourcing host ROS2 environment: ${HOST_ROS_PATH}/setup.bash"
        source "${HOST_ROS_PATH}/setup.bash"
    else
        log_error "宿主机 ROS2 未找到: ${HOST_ROS_PATH}/setup.bash"
        exit 1
    fi

    # --- 环境变量 ---
    export ROS2_INSTALL_PATH=${work_dir}/install
    export SERES_INTERFACE_INSTALL_DIR=${work_dir}/install
    export MID_INSTALL_DIR=${work_dir}/install
    export SYSROOT_CUDA_HOME="${SYSROOT}/usr/local/cuda"
    export PYTHON_SOABI=cpython-310-aarch64-linux-gnu

    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_DISTRO=humble
    export ROS_LOCALHOST_ONLY=0
    export ROS_PACKAGE_PATH=${TARGET_ROS_PATH}/share
    export CMAKE_PREFIX_PATH="${TARGET_ROS_PATH}:${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake:${SYSROOT}/usr/local/lib/cmake:${work_dir}/install"
    export PKG_CONFIG_PATH="${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${SYSROOT}/usr/share/pkgconfig"
    export PKG_CONFIG_SYSROOT_DIR="${SYSROOT}"
    export PYTHON_EXECUTABLE_PATH="/usr/bin/python3"
    export PYTHONPATH="${HOST_ROS_PATH}/local/lib/${PYTHON_VERSION}/dist-packages:${HOST_ROS_PATH}/lib/${PYTHON_VERSION}/site-packages"

    # --- CUDA 配置 ---
    CUDA_HOME="/usr/local/cuda"
    CUDA_HOME_ARM64="${SYSROOT}${CUDA_HOME}"
    export CUDA_HOME=${CUDA_HOME_ARM64}
    export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME_ARM64}/bin
    export CUDA_INCLUDE_DIRS=${CUDA_HOME_ARM64}/include
    export CUDA_LIBRARY_DIRS=${CUDA_HOME_ARM64}/lib64
    export CUDA_CUDART_LIBRARY=${CUDA_HOME_ARM64}/lib64/libcudart.so
    export CUDACXX=${CUDA_HOME_ARM64}/bin/nvcc
    export CUDA_GPU_CC="87"

    export PATH=${CUDA_HOME_ARM64}/bin:${PATH}
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${HOST_ROS_PATH}/opt/rviz_ogre_vendor/lib:${HOST_ROS_PATH}/lib/x86_64-linux-gnu:${HOST_ROS_PATH}/lib:${LD_LIBRARY_PATH}"

    # --- 搜索路径 ---
    export CPATH="${SYSROOT}/usr/local/include:${TARGET_ROS_PATH}/include:${CUDA_HOME_ARM64}/include:${CPATH}"
    export LIBRARY_PATH="${SYSROOT}/usr/lib/aarch64-linux-gnu:${SYSROOT}/usr/local/lib:${LIBRARY_PATH}"

    # --- CMake 参数 ---
    CMAKE_CCP_PARAM="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH}"

    log_info "  SYSROOT_DIR          : ${SYSROOT_DIR}"
    log_info "  TOOLCHAIN_FILE       : ${TOOLCHAIN_FILE}"
    log_info "  TARGET_ARCH          : ${TARGET_ARCH}"
    log_info "  build_type           : ${build_type}"
    log_info "  CUDA_HOME (arm64)    : ${CUDA_HOME_ARM64}"
    log_info "  CMAKE_PREFIX_PATH    : ${CMAKE_PREFIX_PATH}"
    log_info "  HOST_ROS_PATH        : ${HOST_ROS_PATH}"
}

function build_arm() {
    basepath=$(pwd)
    log_info "开始 ARM 交叉编译..."

    # 校验路径
    TOOLCHAIN_FILE=$(realpath "${TOOLCHAIN_FILE}" 2>/dev/null || echo "${TOOLCHAIN_FILE}")
    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        log_error "toolchain 文件不存在: ${TOOLCHAIN_FILE}"
        exit 1
    fi
    if [ ! -d "${SYSROOT_DIR}" ]; then
        log_error "sysroot 目录不存在: ${SYSROOT_DIR}"
        exit 1
    fi

    init_cross_compile_env
    
    
    log_info "编译 kygi665_msgs rhr_interface nav2_interface  common_interface 和 novatel_oem7_msgs..."
    colcon build --base-paths ../seres_interface/ \
        --packages-select  kygi665_msgs rhr_interface nav2_interface common_interface  \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH="${TARGET_ROS_PATH};${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/local/lib/cmake;${work_dir}/install" \
            ${CMAKE_CCP_PARAM}
    colcon build \
        --packages-select  novatel_oem7_msgs \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH="${TARGET_ROS_PATH};${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/local/lib/cmake;${work_dir}/install" \
            ${CMAKE_CCP_PARAM}

    if [ $? -ne 0 ]; then
        log_error " kygi665_msgs rhr_interface nav2_interface common_interface  novatel_oem7_msgs 交叉编译失败"
        exit 1
    fi

    log_ok " kygi665_msgs rhr_interface nav2_interface common_interface  novatel_oem7_msgs 编译完成！开始编译 basic..."
    sleep 3

    source install/setup.bash

    log_info "编译 basic..."
    colcon build  \
        --packages-select basic \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH="${TARGET_ROS_PATH};${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/local/lib/cmake;${work_dir}/install" \
            ${CMAKE_CCP_PARAM}

    if [ $? -ne 0 ]; then
        log_error "basic 交叉编译失败"
        exit 1
    fi

    log_ok "basic 编译完成！开始编译 livox_ros_driver2..."
    sleep 3

    source install/setup.bash

    log_info "编译 livox_ros_driver2..."
    colcon build  \
        --packages-select livox_ros_driver2 \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH="${TARGET_ROS_PATH};${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/local/lib/cmake;${work_dir}/install" \
            ${CMAKE_CCP_PARAM}

    if [ $? -ne 0 ]; then
        log_error "livox_ros_driver2 交叉编译失败"
        exit 1
    fi

    log_ok "livox_ros_driver2 编译完成！开始编译 lightning..."
    sleep 3
    
    source install/setup.bash

    log_info "编译 lightning..."
    
    HOST_PROTOC=$(which protoc)
    if [ -z "$HOST_PROTOC" ]; then
        log_error "Host protoc not found."
        exit 1
    fi
    log_info "Using host protoc: $HOST_PROTOC"

    colcon build \
        --packages-select lightning \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH="${TARGET_ROS_PATH};${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake;${SYSROOT}/usr/local/lib/cmake;${work_dir}/install" \
            -DCMAKE_MODULE_PATH="${SYSROOT}/usr/share/cmake/geographiclib;${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake" \
            -DProtobuf_PROTOC_EXECUTABLE=${HOST_PROTOC} \
            -DCMAKE_CXX_FLAGS="-Wno-error -Wno-unused-parameter -Wno-sign-compare -Wno-reorder" \
            ${CMAKE_CCP_PARAM}

    if [ $? -ne 0 ]; then
        log_error "lightning 交叉编译失败"
        exit 1
    fi
    
    log_ok "lightning 编译完成！"
    echo "arm64" > $basepath/.build_arch
    
    # 清理产物中的 sysroot 绝对路径
    log_info "清理 install 目录中 cmake 文件内的 sysroot 绝对路径..."
    find ${basepath}/install -name "*.cmake" -exec sed -i "s|${SYSROOT_DIR}||g" {} \;
    
    unset AMENT_PREFIX_PATH
}

function build_x86() {
    basepath=$(pwd)
    log_info "开始 x86 编译..."
    echo $basepath
    source /opt/ros/humble/setup.bash
    

    
    log_info "编译  kygi665_msgs rhr_interface nav2_interface common_interface 和 novatel_oem7_msgs..."
    colcon build \
        --packages-select kygi665_msgs rhr_interface nav2_interface common_interface  novatel_oem7_msgs

    if [ $? -ne 0 ]; then
        log_error " kygi665_msgs rhr_interface nav2_interface common_interface  novatel_oem7_msgs 编译失败"
        exit 1
    fi

    log_ok " kygi665_msgs  rhr_interface nav2_interface common_interface novatel_oem7_msgs 编译完成！开始编译 lightning..."
    sleep 3

    source install/setup.bash

    log_info "编译 lightning..."
    colcon build \
        --packages-select lightning \
        --cmake-args \
            -DCMAKE_PREFIX_PATH=/opt/ros/humble

    if [ $? -ne 0 ]; then
        log_error "lightning 编译失败"
        exit 1
    fi

    log_ok "lightning 编译完成！"
    echo "amd64" > $basepath/.build_arch
}


function package() {
    basepath=$(pwd)
    echo $basepath
    echo "package slam..."

    local VERSION_NUMBER="1.0.1"
    local PACKAGE_NAME="slam"
    local DEB_PACKAGE_NAME="slam"
    rm -rf $basepath/*.deb

    if [ -z "${robot_type}" ]; then
        robot_type="D20"
    fi

    if [[ -z "$CI_COMMIT_REF_NAME" || -z "$CI_COMMIT_SHA" ]]; then
        CI_COMMIT_REF_NAME=$(git branch --show-current 2>/dev/null || echo "unknown")
        CI_COMMIT_SHA=$(git log -1 --format=%H 2>/dev/null || echo "unknown")
    fi

    declare -A archs
    archs["x86"]="amd64"
    archs["aarch"]="arm64"
    archs["aarch64"]="arm64"
    archs["arm64"]="arm64"
    archs["x86_aarch"]="arm64"

    local ARCH=""
    if [ -f "$basepath/.build_arch" ]; then
        ARCH=$(cat $basepath/.build_arch)
        echo "Detected build architecture from .build_arch: $ARCH"
    elif [ -z "${CI_ARCH}" ] || [[ ! ${archs[$CI_ARCH]} ]]; then
        ARCH="amd64"
    else
        ARCH=${archs[${CI_ARCH}]}
    fi

    local TIMESTAMP=$(date +"%Y%m%d%H")
    CI_COMMIT_REF_NAME="${CI_COMMIT_REF_NAME##*/}"
    local VERSION_STRING=v${VERSION_NUMBER}-${TIMESTAMP}
    local DEB_DIR="build_tmps/${PACKAGE_NAME}-${robot_type}-Humble-${VERSION_STRING}-${ARCH}"
    echo "deb package name: ${DEB_DIR}"

    mkdir -p "${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}"

    if [[ -d "install" ]]; then
        cp -r install/* ${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}
        if [[ "${ARCH}" == "arm64" ]]; then
            find "${DEB_DIR}" -type f \( -name "setup.bash" -o -name "local_setup.bash" -o -name "local_setup.sh" -o -name "setup.sh" -o -name "_local_setup_util_sh.py" -o -name "*.cmake" \) -exec sed -i 's|/opt/aarch64/opt/ros/humble|/opt/ros/humble|g' {} \;
        fi
    else
        echo "install/ directory not found, abort."
        rm -rf "build_tmps"
        exit 1
    fi

    cat <<EOF >"${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}/version.txt"
date: $(date +"%Y-%m-%d %H:%M:%S")
sha: ${CI_COMMIT_SHA:0:6}
branch: ${CI_COMMIT_REF_NAME}
EOF

    mkdir -p "${DEB_DIR}/DEBIAN"

    cat <<EOF >"${DEB_DIR}/DEBIAN/control"
Package: ${DEB_PACKAGE_NAME}
Version: ${VERSION_NUMBER}
Architecture: ${ARCH}
Maintainer: <tian.zhao@seres.cn>
Description: ${PACKAGE_NAME}. Installs to /opt/seres/nav/${PACKAGE_NAME}
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/preinst"
#!/bin/bash
if [ -d "/opt/seres/nav/${PACKAGE_NAME}" ]; then
    systemctl stop ${PACKAGE_NAME}.service 2>/dev/null || true
    systemctl disable ${PACKAGE_NAME}.service 2>/dev/null || true
    rm -f /etc/systemd/system/${PACKAGE_NAME}.service
    systemctl daemon-reload
    rm -rf /opt/seres/nav/${PACKAGE_NAME}
fi
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/postinst"
#!/bin/bash
cat > /etc/systemd/system/${PACKAGE_NAME}.service << SVCEOF
[Unit]
Description=seres slam service
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/seres/nav/${PACKAGE_NAME}
ExecStart=/bin/bash -c 'source /opt/seres/nav/${PACKAGE_NAME}/setup.bash; ros2 launch lightning model_switch_launch.py'
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=slam

[Install]
WantedBy=multi-user.target
SVCEOF
systemctl daemon-reload
systemctl enable ${PACKAGE_NAME}.service
systemctl start ${PACKAGE_NAME}.service
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/prerm"
#!/bin/bash
systemctl stop ${PACKAGE_NAME}.service 2>/dev/null || true
systemctl disable ${PACKAGE_NAME}.service 2>/dev/null || true
rm -f /etc/systemd/system/${PACKAGE_NAME}.service
systemctl daemon-reload
rm -rf /opt/seres/nav/${PACKAGE_NAME}
EOF

    cat ${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}/version.txt || true

    chmod 755 "${DEB_DIR}/DEBIAN/postinst"
    chmod 755 "${DEB_DIR}/DEBIAN/preinst"
    chmod 755 "${DEB_DIR}/DEBIAN/prerm"
    chmod 644 "${DEB_DIR}/DEBIAN/control"
    find "${DEB_DIR}" -type d -exec chmod 755 {} \;

    echo "DEB_DIR: ${DEB_DIR}"
    dpkg-deb -Zgzip -z6 --build "${DEB_DIR}"

    if [ ! -f "${DEB_DIR}.deb" ]; then
        echo "deb build failed"
        rm -rf "build_tmps"
        exit 1
    fi

    cp "${DEB_DIR}.deb" "$basepath"
    rm -rf "build_tmps"

    echo "deb package built at $basepath"
    echo "package done."
}

function download_and_install() {
    local host=$1
    if [ -z "$host" ]; then
        echo "Error: target host not specified"
        exit 1
    fi

    local deb_file=$(ls -t ${work_dir}/*.deb 2>/dev/null | head -1)
    if [ -z "$deb_file" ]; then
        echo "Error: no .deb file found in ${work_dir}"
        exit 1
    fi

    local deb_name=$(basename $deb_file)
    local remote_tmp="/tmp/${deb_name}"

    echo "Deploying ${deb_name} to ${remote_user}@${host}..."

    sshpass -p "${remote_pwd}" scp ${ssh_options} "${deb_file}" "${remote_user}@${host}:${remote_tmp}"
    if [ $? -ne 0 ]; then
        echo "Error: scp failed"
        exit 1
    fi

    sshpass -p "${remote_pwd}" ssh ${ssh_options} "${remote_user}@${host}" \
        "echo '${remote_pwd}' | sudo -S dpkg -i ${remote_tmp} && rm -f ${remote_tmp}"
    if [ $? -ne 0 ]; then
        echo "Error: dpkg install failed"
        exit 1
    fi

    echo "Deployment to ${host} completed"
}

function print_usage() {
    echo "build_l4t.sh 用法:"
    echo "  -arm [-s <sysroot路径>] [-t <toolchain路径>] [-b <Debug|Release|RelWithDebInfo>]"
    echo "                              交叉编译 ARM 版本（可指定 sysroot、toolchain 和构建类型）"
    echo "  -x86                        编译 x86 版本"
    echo "  -p                          打包成 .deb 安装包"
    echo "  -deploy <host>              部署到指定 Jetson 设备"
    echo ""
    echo "示例:"
    echo "  ./build_l4t.sh -arm"
    echo "  ./build_l4t.sh -arm -s /custom/sysroot -t /custom/toolchain.cmake -b Debug"
    echo "  ./build_l4t.sh -x86"
    echo "  ./build_l4t.sh -p"
    echo "  ./build_l4t.sh -deploy 192.168.1.100"
}

function main() {
    if [ $# -lt 1 ]; then
        print_usage
        exit
    fi

    platform=$1
    shift

    case $platform in
    -arm)
        while [[ $# -gt 0 ]]; do
            case $1 in
                -s) SYSROOT_DIR="$2"; shift 2 ;;
                -t) TOOLCHAIN_FILE="$2"; shift 2 ;;
                -b) build_type="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        build_arm
        ;;
    -x86)
        build_x86
        ;;
    -p|-package)
        package
        ;;
    -deploy)
        if [ -z "$1" ]; then
            log_error "请指定目标主机地址"
            exit 1
        fi
        download_and_install "$1"
        ;;
    *)
        print_usage
        ;;
    esac
}

main $@
