#!/bin/bash
# 用 /public 上的 x86 Humble 前缀启动已编好的 planning，给本机回放或联调感知/定位。
# 狗上的正式进程是 planning.service，不要用这个脚本替代。

set -euo pipefail

prefix=/public/zjst/robot/planning_humble_env
env_prefix="${prefix}/env"
install_base="${prefix}/colcon/install"
bin="${install_base}/planning/lib/planning/planning"
setup="${install_base}/setup.bash"

if [[ ! -x "${bin}" || ! -f "${setup}" ]]; then
  echo "planning 还没有 x86 编译产物。"
  echo "缺少: ${bin}"
  echo "说明见 docs/ONBOARDING.md 里「在这台机器上跑起来」。"
  exit 1
fi

export MAMBA_ROOT_PREFIX="${prefix}/root"
export CONDA_BUILD=0
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="${env_prefix}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${env_prefix}"
# shellcheck disable=SC1090
source "${setup}"
set -u

if [[ -z "${PLANNING_ROOT_DIR:-}" ]]; then
  echo "PLANNING_ROOT_DIR 没有被 install/setup.bash 导出。"
  exit 1
fi

# 车型：PLANNING_VIN，默认 d20/001（新版机器人团队的默认是 d30/001，T48）
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/planning_vin.sh"
PLANNING_ROOT_DIR="$(planning_root_for_vin "${PLANNING_ROOT_DIR}" "${PLANNING_VIN:-d20/001}" /tmp/planning_debug_root)" \
  || exit 1
export PLANNING_ROOT_DIR
vin_file="${PLANNING_ROOT_DIR}/robot_factory/default_robot_vin"
echo "PLANNING_ROOT_DIR=${PLANNING_ROOT_DIR}"
echo "vin=$(tr -d '[:space:]' < "${vin_file}")"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未设置}"
echo "日志目录 /tmp ，节点名 planning / planning_debug_bridge"
echo "D20 默认不发调试话题。要在 RViz 看路径，把 communication_conf.pb.txt 里 pub_flag 改成 true。"

exec "${bin}"
