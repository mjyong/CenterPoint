#!/bin/bash
# 用 /public 上的 x86 Humble 前缀启动 grid_map_memory_node，发布规划用的局部栅格。
# 狗上的正式进程不要用这个脚本替代。

set -euo pipefail

prefix=/public/zjst/robot/planning_humble_env
env_prefix="${prefix}/env"
install_base="${prefix}/colcon/install"
bin="${install_base}/grid_map_memory_node/lib/grid_map_memory_node/grid_map_memory_node"
setup="${install_base}/setup.bash"

if [[ ! -x "${bin}" || ! -f "${setup}" ]]; then
  echo "grid_map_memory_node 还没有 x86 编译产物。"
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
echo "D20 配置是 lidar_only，订阅 /patchworkpp/ground_seg_intensity，发布 /map/occupancy_grid。"

exec "${bin}"
