#!/bin/bash
script_abs_file_path=$(realpath "$0")
script_abs_dir_path=$(dirname ${script_abs_file_path})
export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path
export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install
echo "SERES_PERCEPTION_ROOT_DIR: $SERES_PERCEPTION_ROOT_DIR"
export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/install/glog-0.6.0/lib:${SERES_PERCEPTION_ROOT_DIR}/install/octomap/lib:${LD_LIBRARY_PATH}
cd ${SERES_PERCEPTION_ROOT_DIR}/install/lidar_3d_detection/lib/lidar_3d_detection
source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash
./lidar_3d_detection
