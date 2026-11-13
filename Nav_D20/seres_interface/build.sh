#!/bin/bash

# FLUTTER
#export PUB_HOSTED_URL=https://pub.flutter-io.cn
#export FLUTTER_STORAGE_BASE_URL=https://storage.flutter-io.cn
#export FLUTTER_SDK=/root/cc_ws/flutter-elinux/flutter
#export PATH="$PATH:/root/cc_ws/flutter-elinux/bin:$FLUTTER_SDK/bin"
#export ELINUX_ENGINE_BASE_LOCAL_DIRECTORY=/root/cc_ws/flutter-elinux/flutter/bin/cache/downloads/github.com/sony/flutter-embedded-linux/releases/download/db49896cf2

set -e

arch="x86_aarch"
fail_continue="false"
fail_param=""
curdir=$(dirname "$(realpath "$0")")
sysroot=""
packages=""
up_to_packages=""
clean="false"
merge_install="false"
debug_sym="false"
skip_packages=""
skip_default="false"  # 新增标志变量

DEFAULT_DEPENDENCY="seres_default_tmp_dep"
CI_OPT_DEPENDENCY="${CI_OPT_DEPENDENCY:-$DEFAULT_DEPENDENCY}"
echo "[INFO] CI_OPT_DEPENDENCY=${CI_OPT_DEPENDENCY}"
print_help() {
    length=${#1}
    spaces=$(printf "%*s" "$length" "")
    echo "$1 [--arch <x86 | aarch | x86_aarch>] 定目标产物CPU架构，不指定默认交叉编译"
    echo "${spaces} [--sysroot </xx/xx/sysroot_docker>] 指定交叉编译根目录"
    echo "${spaces} [--fail-continue | -f] 允许某些包失败后继续编译"
    echo "${spaces} [--help | -h] 查看帮助信息"
    echo "${spaces} [--packages-select | -p <package1 packages2...>] 指定某些包编译（仅构建指定包）"
    echo "${spaces} [--packages-up-to | -s <package1 packages2...>] 指定某些包编译（构建指定包及其依赖）"
    echo "${spaces} [--skip-packages | -k [package1 ...]] 指定要跳过的包，不带参数时默认跳过 robot_webctr"
    echo "${spaces} [--clean | --clear | -c] 清除所有编译产物"
}

while [[ $# -gt 0 ]]; do
    case $1 in
    --arch)
        arch=$2
        shift 2
        ;;
    --sysroot)
        sysroot=$2
        shift 2
        ;;
    --fail-continue | -f)
        fail_continue="true"
        shift
        ;;
    --help | -h)
        print_help $0
        exit 0
        ;;
    --packages-select | -p)
        shift
        packages=$@
        shift $#
        ;;
    --packages-up-to | -s)
        shift
        up_to_packages=$@
        shift $#
        ;;
    --clean | --clear | -c)
        clean="true"
        shift
        ;;
    --skip-packages | -k)
        shift
        # 检查是否有参数
        if [ $# -gt 0 ] && [[ $1 != -* ]]; then
            skip_packages=$@
            shift $#
        else
            # 没有参数，设置默认跳过 robot_webctr
            skip_default="true"
        fi
        ;;
    *)
        echo "illegal input $*"
        print_help $0
        exit 1
        ;;
    esac
done

if [ "$arch" = "humble_x86_aarch" ]; then
    arch="x86_aarch"
fi
if [ "$arch" = "humble_x86" ]; then
    arch="x86"
fi

declare -A archs
archs["x86"]="1"
archs["aarch"]="1"
archs["aarch64"]="1"
archs["x86_aarch"]="1"

if [[ ! ${archs[$arch]} ]]; then
    echo "${arch} not in ${!archs[@]}"
    exit 1
fi

if [ -z "${sysroot}" ] && [ "x${arch}" == "xx86_aarch" ]; then
    echo "you must input sysroot for ${arch}"
    exit 1
fi

if [ "x${clean}" == "xtrue" ]; then
    dirs=(install/ build/ log/ src/module/robot_common/include/proto/ tmp/)
    for dir in ${dirs[@]}; do
        echo "rm -rf $dir"
        rm -rf $dir
    done
fi

TOOLCHAIN_FILE="${curdir}/generic_linux.cmake"
CMAKE_EXTRA_ARGS=""
BUILD_TYPE="Debug"

if [ "x${SOC_PLATFORM}" == "xQCOM_6490" ]; then
    echo "set environment variables for QCOM_6490 Android NDK cross-compilation"
    NDK_PATH="${sysroot:-${ANDROID_NDK}}"
    TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake"
    BUILD_TYPE="Release"
    CMAKE_EXTRA_ARGS="-DBUILD_TESTING=OFF -DROSIDL_GENERATOR_PY=OFF -DRCL_LOGGING_IMPLEMENTATION=rcl_logging_noop -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DANDROID_STL=c++_shared -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=65536"
elif [ "x${arch}" == "xx86_aarch" ]; then
    echo "set environment variables for cross-compilation"
    export TARGET_ARCH=aarch64
    export TARGET_TRIPLE=aarch64-linux-gnu
    if [ "x${ROBOT_TYPE}" == "xD30" ]; then
        export CC=$TARGET_TRIPLE-gcc
        export CXX=$TARGET_TRIPLE-g++
        TOOLCHAIN_FILE="${curdir}/d30_linux.cmake"
    else
        CC_WS=/root/cc_ws
        export CC=/usr/bin/$TARGET_TRIPLE-gcc
        export CXX=/usr/bin/$TARGET_TRIPLE-g++
        export CROSS_COMPILE=/usr/bin/$TARGET_TRIPLE-
    fi
    export SYSROOT=${sysroot}
    export PYTHON_SOABI=cpython-310m-$TARGET_TRIPLE
    export ROS2_INSTALL_PATH=$(pwd)/install
fi

if [ "x${SOC_PLATFORM}" == "xQCOM_6490" ]; then
    unset AMENT_PREFIX_PATH
    unset CMAKE_PREFIX_PATH
    unset COLCON_PREFIX_PATH
    unset LD_LIBRARY_PATH
    unset PYTHONPATH
    unset ROS_DISTRO
    unset ROS_VERSION
    export PATH=$(echo $PATH | sed 's#:/opt/ros/humble/bin##g' | sed 's#/opt/ros/humble/bin:##g' | sed 's#/opt/ros/humble/bin##g')

    source /opt/ros2/ros2_host/install/setup.bash
    source /opt/ros2/ros2_android/install/setup.bash
elif [ -f ${sysroot%/}/opt/ros/humble/setup.bash ]; then
    . ${sysroot%/}/opt/ros/humble/setup.bash
fi
#if [ -f /opt/ros/foxy/setup.bash ]; then
#    . ${sysroot}/opt/ros/foxy/setup.bash
#fi

umask 0000
# echo 3 > /proc/sys/vm/drop_caches || true
# echo 3 > /proc1/sys/vm/drop_caches || true

if [ "x${fail_continue}" == "xtrue" ]; then
    fail_param="--continue-on-error"
fi

# Get the total available memory in KB from /proc/meminfo
mem_gb=$(free -g | awk '/^Mem:/ {print $4}')
cpu_cores=$(nproc)
# 使用核心数的1/4，至少保留1个核心
cpu_cores_quarter=$((cpu_cores / 4))
cpu_cores_quarter=$((cpu_cores_quarter > 0 ? cpu_cores_quarter : 1))
mem_concurrency=$((mem_gb * 2 / 5))
concurrency=$((mem_concurrency < cpu_cores_quarter ? mem_concurrency : cpu_cores_quarter))
concurrency=$((concurrency > 0 ? concurrency : 1))

echo "arch:       ${arch}"
echo "sysroot:    ${sysroot}"
echo "fail_param: ${fail_param}"
echo "parallel:   ${concurrency}"

start_time=$(date +%s)

build_by_selectnames() {
    echo "-------------------------------"
    echo "$@"
    echo "-------------------------------"
    packages=""
    if [ $# -ge 1 ]; then
        # packages="--packages-up-to $@"
        packages="--packages-select $@"
    fi
    if [ "x${arch}" == "xx86_aarch" ]; then
        colcon build --merge-install --parallel-workers ${concurrency} \
        --cmake-force-configure --cmake-args ${BUILD_PROJECT} \
         -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        ${CMAKE_EXTRA_ARGS} ${fail_param} ${packages}
    else
        colcon build --merge-install --parallel-workers ${concurrency} \
        --cmake-force-configure --cmake-args ${BUILD_PROJECT} \
         -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ${fail_param} ${packages}
    fi
}

build_by_names() {
    echo "-------------------------------"
    echo "$@"
    echo "-------------------------------"
    packages=""
    if [ $# -ge 1 ]; then
        packages="--packages-up-to $@"
        # packages="--packages-select $@"
    fi
    if [ "x${arch}" == "xx86_aarch" ]; then
        colcon build --merge-install --parallel-workers ${concurrency} \
        --cmake-force-configure --cmake-args ${BUILD_PROJECT} \
         -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        ${CMAKE_EXTRA_ARGS} ${fail_param} ${packages}
    else
        colcon build --merge-install --parallel-workers ${concurrency} \
        --cmake-force-configure --cmake-args ${BUILD_PROJECT} \
         -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ${fail_param} ${packages}
    fi
}

if [ ! -z "${packages}" ] || [ ! -z "${up_to_packages}" ]; then
    if [ -f "install/setup.bash" ]; then
        . install/setup.bash
    fi
    if [ ! -z "${packages}" ]; then
        build_by_selectnames "${packages}"
    elif [ ! -z "${up_to_packages}" ]; then
        build_by_names "${up_to_packages}"
    fi
else
    names=" "
    names+="seres_proto "
    names+="audio_interface "
    names+="hmi_interface "
    names+="light_interface "
    names+="robot_interface "
    names+="scene_interface "
    names+="power_interface "
    names+="log_interface "
    names+="media_interface "
    names+="iot_interface "
    names+="mc_interface "
    names+="rte_interface "
    names+="tau_interface "
    names+="sysapp_interface "
    names+="voice_interface "
    names+="rhr_interface "
    names+="profile_interface "
    names+="nav2_interface "
    names+="calibration_interface "
    names+="perception_interface "
    names+="device_info_interface "
    names+="manipulation_interface "
    names+="shm_msgs "
    names+="agv_interfaces "
    if [ "x${ROBOT_TYPE}" == "xR11-W" ]; then
        names+="woosh_action_msgs "
        names+="woosh_common_msgs "
        names+="woosh_nav_msgs "
        names+="woosh_robot_msgs "
        names+="woosh_ros_msgs "
        names+="woosh_task_msgs "
        names+="woosh_map_msgs "
    elif [ "x${ROBOT_TYPE}" == "xD20" ]; then
        names+="zsi_interface "
        names+="kygi665_msgs "
    fi

    if [ ! -z "${names}" ]; then
        echo -e "\e[47;30m build normal packages \e[0m"
        build_by_names ${names}
    fi
fi

declare -A soabis
soabis["x86"]="cpython-310-x86_64-linux-gnu"
soabis["aarch"]="cpython-310m-aarch64-linux-gnu"
soabis["aarch64"]="cpython-310m-aarch64-linux-gnu"
soabis["x86_aarch"]="cpython-310m-aarch64-linux-gnu"
find "install/" -type f -name "*.${soabis[${arch}]}.so" | while read -r file; do
new_file=$(echo "$file" | sed "s/\\.${soabis[${arch}]}//")
if [ -f "${file}" ]; then
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
fi
done

end_time=$(date +%s)
elapsed_time=$(($end_time - $start_time))
echo "total cost time of compile: $elapsed_time s"

start_time=$(date +%s)
tmp_install_dir="${sysroot}/opt/${CI_OPT_DEPENDENCY}/${arch}/seres_interface"
mkdir -p ${tmp_install_dir}
rm -rf ${tmp_install_dir}/*
cp -rf ${curdir}/install ${tmp_install_dir}/
end_time=$(date +%s)
elapsed_time=$(($end_time - $start_time))
echo "copy cost time of compile: $elapsed_time s"
