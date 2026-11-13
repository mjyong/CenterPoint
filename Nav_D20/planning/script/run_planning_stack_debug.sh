#!/bin/bash
# 用 lidar_3d_detection 同一份 bag，一起拉起地面分割、局部栅格、定位和规划。
# 调试规划用。会发 /cmd_vel，默认 ROS_DOMAIN_ID=42，不要和狗用同一个域。

set -euo pipefail

prefix=/public/zjst/robot/planning_humble_env
env_prefix="${prefix}/env"
install_base="${prefix}/colcon/install"
nav_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# bag 和 Nav_D20 并列：在本仓库是 d20car_predict/bags0921，放进 CenterPoint 后是仓库根上的 bags0921。
default_bag="$(cd "${nav_root}/.." && pwd)/bags0921/rosbag2_2026_09_21-16_41_27"
loc_config="${nav_root}/slam/lightning-lm-master/config/default_d20.yaml"
geo_lib="${nav_root}/slam/lightning-lm-master/thirdparty/geographiclib/X86/lib"
log_dir=/tmp/nav_stack_debug

bag="${default_bag}"
loop=1
with_planning=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) loop=0; shift ;;
    --no-planning) with_planning=0; shift ;;
    --bag) bag="$2"; shift 2 ;;
    -h|--help)
      echo "用法: bash $0 [--bag BAG] [--once] [--no-planning]"
      echo "默认 bag: ${default_bag}"
      exit 0
      ;;
    *) bag="$1"; shift ;;
  esac
done

lidar_bin="${install_base}/lidar_seg/lib/lidar_seg/lidar_seg"
grid_bin="${install_base}/grid_map_memory_node/lib/grid_map_memory_node/grid_map_memory_node"
loc_bin="${install_base}/lightning/lib/lightning/run_loc_online"
plan_bin="${install_base}/planning/lib/planning/planning"
for f in "${lidar_bin}" "${grid_bin}" "${loc_bin}" "${bag}/metadata.yaml" "${loc_config}"; do
  if [[ ! -e "${f}" ]]; then
    echo "缺少: ${f}"
    exit 1
  fi
done
if [[ "${with_planning}" -eq 1 && ! -x "${plan_bin}" ]]; then
  echo "缺少: ${plan_bin}"
  exit 1
fi

export MAMBA_ROOT_PREFIX="${prefix}/root"
export CONDA_BUILD=0
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export SERES_PERCEPTION_INSTALL_DIR="${install_base}"
# LOC_MAP_PATH=<定位地图目录>、LOC_INS_ONLY=1：换定位地图 / INS 直出，见 lightning 的 scripts/loc_debug_config.sh
# LOC_INS_YAW_OFFSET：INS 朝向偏置，没设就按 bag 的录制日期定（0921 那批天线接反 180，1016 起 0；T48 用错了，狗头反了）
# shellcheck disable=SC1091
source "${nav_root}/slam/lightning-lm-master/scripts/loc_debug_config.sh"
LOC_INS_YAW_OFFSET="${LOC_INS_YAW_OFFSET:-$(loc_ins_yaw_offset_for_bag "${bag}")}"
export LOC_INS_YAW_OFFSET
echo "INS 朝向偏置 ${LOC_INS_YAW_OFFSET:-配置里的值}（bag $(basename "${bag}")）"
loc_config="$(loc_debug_config "${loc_config}")" || exit 1
export config_path="${loc_config}"
export PLANNING_TIMEOUT_WALL_CLOCK=1
export LD_LIBRARY_PATH="${geo_lib}:${env_prefix}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${env_prefix}"
# shellcheck disable=SC1090
source "${install_base}/setup.bash"
set -u
# 车型：PLANNING_VIN，默认 d20/001（新版机器人团队的默认是 d30/001，T48）
# shellcheck disable=SC1091
source "${nav_root}/planning/script/planning_vin.sh"
PLANNING_ROOT_DIR="$(planning_root_for_vin "${PLANNING_ROOT_DIR}" "${PLANNING_VIN:-d20/001}" "${log_dir}/planning_root")" \
  || exit 1
# PLANNING_STARTUP_MAP=<站点地图目录>：规划起来就读这张图和人行横道（startup_map_dir，狗上就这么配），不用 MAP_SWITCH
if [[ -n "${PLANNING_STARTUP_MAP:-}" ]]; then
  PLANNING_ROOT_DIR="$(planning_root_with_startup_map "${PLANNING_ROOT_DIR}" "${PLANNING_VIN:-d20/001}" \
    "${PLANNING_STARTUP_MAP}" "${log_dir}/planning_root_map")" || exit 1
  echo "规划起来就读的地图：${PLANNING_STARTUP_MAP}"
fi
export PLANNING_ROOT_DIR
echo "规划车型 ${PLANNING_VIN:-d20/001}（PLANNING_ROOT_DIR=${PLANNING_ROOT_DIR}）"

mkdir -p "${log_dir}"
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]+"${pids[@]}"}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "bag=${bag}"
echo "日志 ${log_dir}"
echo "链路: bag 的顶激光/IMU/INSPVAX -> lidar_seg -> /map/occupancy_grid，定位 -> /lightning/odom -> planning"

"${lidar_bin}" >"${log_dir}/lidar_seg.log" 2>&1 &
pids+=($!)
"${grid_bin}" >"${log_dir}/grid_map.log" 2>&1 &
pids+=($!)
"${loc_bin}" >"${log_dir}/loc.log" 2>&1 &
pids+=($!)
sleep 3

# --clock 让 /clock 跟着 bag 时间走。RViz 必须开 use_sim_time，否则 9 月的 TF 会被当成过期数据丢掉。
play_args=(--rate 1 --clock --read-ahead-queue-size 500 --delay 1)
if [[ "${loop}" -eq 1 ]]; then
  play_args+=(--loop)
fi
# BAG_PLAY_EXTRA：多给 ros2 bag play 的参数，例如把狗上录的感知目标改名避开、换成现场重算的：
#   BAG_PLAY_EXTRA="--remap /perception/lidar_3d/objects:=/bag/perception/lidar_3d/objects"
# PERCEPTION_CMD：和整套一起起的感知节点（我们的 lidar_3d_detection，发 /perception/lidar_3d/objects），日志 perception.log
if [[ -n "${BAG_PLAY_EXTRA:-}" ]]; then
  read -r -a extra <<< "${BAG_PLAY_EXTRA}"
  play_args+=("${extra[@]}")
fi
if [[ -n "${PERCEPTION_CMD:-}" ]]; then
  bash -c "${PERCEPTION_CMD}" >"${log_dir}/perception.log" 2>&1 &
  pids+=($!)
  echo "感知节点：${PERCEPTION_CMD}"
fi
ros2 bag play "${play_args[@]}" "${bag}" >"${log_dir}/bag.log" 2>&1 &
bag_pid=$!
pids+=("${bag_pid}")

if [[ "${with_planning}" -eq 1 ]]; then
  sleep 2
  "${plan_bin}" >"${log_dir}/planning.log" 2>&1 &
  pids+=($!)
fi

echo "已启动 pid: ${pids[*]}"
echo "Ctrl-C 会停掉 bag、感知、定位和规划。"
if [[ "${loop}" -eq 0 ]]; then
  # --once：bag 放完就收（定位、分割、规划不会自己退；以前在这里一直等，录包会多录一段没有输入的，T43）
  wait "${bag_pid}" || true
  sleep 3
  echo "bag 放完了，停掉其余进程"
  exit 0
fi
wait
