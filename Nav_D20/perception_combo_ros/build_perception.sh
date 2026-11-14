#!/bin/bash
# 0. export基础环境
script_abs_file_path=$(realpath "$0")
script_abs_dir_path=$(dirname ${script_abs_file_path})
export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path
export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install
export ON_ORIN_DEBUG=OFF
echo "perception build script path : $script_abs_file_path"
echo "SERES_PERCEPTION_ROOT_DIR: $SERES_PERCEPTION_ROOT_DIR"

# source ROS2 环境（Docker 容器内不会自动加载 ~/.bashrc，需要显式 source）
ROS_SETUP_CANDIDATES=(
    "/opt/ros/humble/setup.bash"
    "/opt/ros/foxy/setup.bash"
    "/opt/ros/galactic/setup.bash"
    "/opt/ros/iron/setup.bash"
)
for ros_setup in "${ROS_SETUP_CANDIDATES[@]}"; do
    if [ -f "$ros_setup" ]; then
        echo "Sourcing ROS2 environment: $ros_setup"
        source "$ros_setup"
        break
    fi
done

if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS2 environment not found! Please install ROS2 or check /opt/ros/."
    exit 1
fi


clean_build() {
    echo "开始清理构建文件..."

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
    
    echo "所有构建文件清理完成!"
    exit 0
}


# 处理命令行参数
while getopts "dhc" opt; do
    case $opt in
        d)
            clean_build
            ;;
        c)
            CROSS_CP=ON

            ;;
    esac
done


colcon_compile_node=(
   visual_perception
   pedestrian_code_conduct
  lidar_seg
  grid_map_memory_node
  ptz_data_detection
  perception_recharge
  lidar_3d_detection
)

# 1. 编译 gflags
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/gflags
# rm -rf build
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/gflags \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_TESTING=OFF
make -j$(nproc) && make install

echo "============================ compile gflags down !============================ "

# 2. 编译log三方库glog
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/glog-0.6.0
# rm -rf build
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/glog-0.6.0 \
  -DWITH_GFLAGS=ON \
  -DCMAKE_PREFIX_PATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags \
  -DCMAKE_INSTALL_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags/lib \
  -DCMAKE_BUILD_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/gflags/lib \
  -DWITH_UNWIND=OFF
make -j$(nproc) && make install

echo "============================ compile glog down !============================ "

# 3. 编译 octomap 库
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related/octomap
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${SERES_PERCEPTION_INSTALL_DIR}/octomap \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_OCTOVIS_SUBPROJECT=OFF \
  -DCMAKE_INSTALL_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap/lib \
  -DCMAKE_BUILD_RPATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap/lib
make -j$(nproc) && make install

echo "============================ compile octomap down !============================ "

# 4. 编译 pcl_msgs 和 octomap_msgs 消息包
cd ${SERES_PERCEPTION_ROOT_DIR}/third_party/octomap_related
colcon build --paths pcl_msgs octomap_msgs \
             --install-base ${SERES_PERCEPTION_INSTALL_DIR} \
             --cmake-args \
                -DCMAKE_PREFIX_PATH=${SERES_PERCEPTION_INSTALL_DIR}/octomap:${CMAKE_PREFIX_PATH}

source ${SERES_PERCEPTION_INSTALL_DIR}/setup.bash

echo "============================ build & source pcl_msgs & octomap_msgs done !============================ "


# 5. 编译并安装 接口common_interface包 下所有的消息包，并 source 环境
cd ${SERES_PERCEPTION_ROOT_DIR}/common_interface/
colcon build \
             --install-base ${SERES_PERCEPTION_INSTALL_DIR}

source ${SERES_PERCEPTION_INSTALL_DIR}/setup.bash

echo "============================ build & source common_interface done !============================ "


# 6. 编译感知所有节点
export COM_IPC=OFF
cd ${SERES_PERCEPTION_ROOT_DIR}/node_src/
colcon build --paths "./*"  --packages-select ${colcon_compile_node[@]} --install-base ${SERES_PERCEPTION_INSTALL_DIR} --cmake-args -DCOM_IPC=${COM_IPC} -DSERES_PERCEPTION_ROOT_DIR=${SERES_PERCEPTION_ROOT_DIR} -DON_ORIN_DEBUG=${ON_ORIN_DEBUG}
