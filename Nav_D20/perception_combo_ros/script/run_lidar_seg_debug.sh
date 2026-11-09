#!/bin/bash
# 用 /public 上的 x86 Humble 前缀启动 lidar_seg，给规划需要的地面分割点云。
# 狗上的正式进程不要用这个脚本替代。

set -euo pipefail

prefix=/public/zjst/robot/planning_humble_env
env_prefix="${prefix}/env"
install_base="${prefix}/colcon/install"
bin="${install_base}/lidar_seg/lib/lidar_seg/lidar_seg"
setup="${install_base}/setup.bash"

if [[ ! -x "${bin}" || ! -f "${setup}" ]]; then
  echo "lidar_seg 还没有 x86 编译产物。"
  echo "缺少: ${bin}"
  exit 1
fi

export MAMBA_ROOT_PREFIX="${prefix}/root"
export CONDA_BUILD=0
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export SERES_PERCEPTION_INSTALL_DIR="${install_base}"
export LD_LIBRARY_PATH="${env_prefix}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${env_prefix}"
# shellcheck disable=SC1090
source "${setup}"
set -u

echo "SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_INSTALL_DIR}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "发布 /patchworkpp/ground_seg_intensity，供 grid_map_memory_node 使用。"

exec "${bin}"
