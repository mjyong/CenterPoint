#!/bin/bash
# 用 /public 上的 x86 Humble 前缀启动定位 run_loc_online。
# 狗上的正式进程不要用这个脚本替代。

set -euo pipefail

prefix=/public/zjst/robot/planning_humble_env
env_prefix="${prefix}/env"
install_base="${prefix}/colcon/install"
bin="${install_base}/lightning/lib/lightning/run_loc_online"
setup="${install_base}/setup.bash"
src_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${src_root}/config/default_d20.yaml"
geo_lib="${src_root}/thirdparty/geographiclib/X86/lib"

if [[ ! -x "${bin}" || ! -f "${setup}" || ! -f "${config_path}" ]]; then
  echo "定位还没有 x86 编译产物，或缺少配置。"
  echo "缺少: ${bin}"
  exit 1
fi

# LOC_MAP_PATH、LOC_INS_ONLY、LOC_INS_YAW_OFFSET（放 0921 的包要 180）：见 loc_debug_config.sh
# shellcheck disable=SC1091
source "${src_root}/scripts/loc_debug_config.sh"
config_path="$(loc_debug_config "${config_path}")" || exit 1

export MAMBA_ROOT_PREFIX="${prefix}/root"
export CONDA_BUILD=0
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export config_path
export LD_LIBRARY_PATH="${geo_lib}:${env_prefix}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${env_prefix}"
# shellcheck disable=SC1090
source "${setup}"
set -u

echo "config_path=${config_path}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "订阅 /robot/topic/rdap/top_point_cloud2 和 /rte/rdap/rtk/imus/data_raw"
echo "发布 /lightning/odom 和 /map。配置里的 /opt/seres 地图不存在时，回退到仓库里的默认地图。"

exec "${bin}"
