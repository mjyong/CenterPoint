#!/bin/bash
# 0. export基础环境
script_abs_file_path=$(realpath "$0")
script_abs_dir_path=$(dirname ${script_abs_file_path})
export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path

export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install
#export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml
echo "perception build script path : $script_abs_file_path"
echo "SERES_PERCEPTION_ROOT_DIR: $SERES_PERCEPTION_ROOT_DIR"
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/install/glog-0.6.0/lib:${SERES_PERCEPTION_ROOT_DIR}/install/octomap/lib:${LD_LIBRARY_PATH}
cd ${SERES_PERCEPTION_ROOT_DIR}/install/pedestrian_code_conduct/lib/pedestrian_code_conduct
source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash
./pedestrian_code_conduct_2d

