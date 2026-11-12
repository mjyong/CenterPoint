#!/bin/bash
# 回放一份 bag 跑整套（run_planning_stack_debug.sh --once），录规划的输入输出，发一个目标点；
# bag 放完以后查录包（tools/check_planning_mcap.py），数规划日志里的关键行。T41–T43 的手工步骤收在这里（T46 复核）。
#
#   bash Nav_D20/planning/script/replay_planning_check.sh <名字> <目标 x> <目标 y> [--bag BAG]
#   例：LOC_MAP_PATH=data/loc_maps/ad68517c-... bash Nav_D20/planning/script/replay_planning_check.sh t47_map 80.0 126.9
#
# 环境变量同 run_planning_stack_debug.sh：LOC_MAP_PATH、LOC_INS_ONLY；ROS_DOMAIN_ID 默认 42（会发 /cmd_vel，不要用狗的域）。
# MAP_SWITCH=<地图目录>（有 global_map.yaml 的）：发目标之前先发 /map/switch，规划换成这张图，读它的 semantic.osm。
# 输出在 work_dirs/planning_io_<名字>/：录包（mcap）、logs/（各进程日志、规划的 glog）、report.txt（查包和关键行）、
# viz/planning_viz.mcap 和 .png（tools/planning_mcap_viz.py：在 Foxglove 里用 script/planning_viz.json 看，不闪）。
# RECORD_CLOUD=1：连 /fused_point_cloud 一起录（大，默认不录）。
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,9p' "$0"
  exit 2
fi
name=$1 goal_x=$2 goal_y=$3
shift 3

nav_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "${nav_root}/.." && pwd)"
out="${repo_root}/work_dirs/planning_io_${name}"
prefix=/public/zjst/robot/planning_humble_env
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export MAMBA_ROOT_PREFIX="${prefix}/root"
export CONDA_BUILD=0
set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${prefix}/env"
# shellcheck disable=SC1091
source "${prefix}/colcon/install/setup.bash"
set -u

rm -rf "${out}"
mkdir -p "$(dirname "${out}")"
stamp="$(date +%s)"
ros2 bag record -s mcap -o "${out}" \
  /lightning/odom /map/occupancy_grid /tf /tf_static \
  /planning/debug/cost_map_production/occmap /planning/debug/global_map_manager/occmap \
  /planning/debug/goal_pose /planning/debug/path_tracking /planning/debug/plotjuggler/string \
  /planning/debug/diagnostic /planning/debug/collision_checker /planning/debug/teb_planner /planning/debug/following \
  /goal_pose /cmd_vel /map/switch $([[ "${RECORD_CLOUD:-0}" == 1 ]] && echo /fused_point_cloud) \
  > "/tmp/replay_${name}_record.txt" 2>&1 &
rec=$!
bash "${nav_root}/planning/script/run_planning_stack_debug.sh" --once "$@" > "/tmp/replay_${name}_stack.txt" 2>&1 &
stack=$!

# 规划起来以后（有调试服务了）：打开调试话题，等录包和规划两个订阅者都连上 /goal_pose 再发目标
# （只等一个的话录包先连上就发了，规划收不到，T43）
sent=0
for _ in $(seq 1 60); do
  if ros2 service list 2>/dev/null | grep -q '^/planning/debug/set_pub_flag$'; then
    ros2 service call /planning/debug/set_pub_flag std_srvs/srv/SetBool "{data: true}" > /dev/null 2>&1 || true
    if [[ -n "${MAP_SWITCH:-}" ]]; then
      timeout 30 ros2 topic pub --once -w 1 /map/switch std_msgs/msg/String "{data: '${MAP_SWITCH}'}" > /dev/null 2>&1 || true
      sleep 3
    fi
    timeout 30 ros2 topic pub --once -w 2 /goal_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: map}, pose: {position: {x: ${goal_x}, y: ${goal_y}, z: 0.0}, orientation: {w: 1.0}}}" \
      > /dev/null 2>&1 && sent=1
    break
  fi
  if ! kill -0 "${stack}" 2>/dev/null; then
    break
  fi
  sleep 1
done
echo "目标 (${goal_x}, ${goal_y}) $([[ ${sent} -eq 1 ]] && echo 已发 || echo 没发出去)"

wait "${stack}" || echo "整套退出码 $?"
kill -INT "${rec}" 2>/dev/null || true
wait "${rec}" 2>/dev/null || true

mkdir -p "${out}/logs"
cp /tmp/nav_stack_debug/*.log "/tmp/replay_${name}_record.txt" "/tmp/replay_${name}_stack.txt" "${out}/logs/" 2>/dev/null || true
plan_log=""
for f in /tmp/planning.*.log.INFO.*; do
  if [[ -f "${f}" && "$(stat -c %Y "${f}")" -ge "${stamp}" ]]; then
    plan_log="${f}"
  fi
done
[[ -n "${plan_log}" ]] && cp "${plan_log}" "${out}/logs/planning.INFO"

# 查包要 numpy、mcap、rosbags：micromamba 的 python3 没有 mcap（T47），优先用仓库的 .venv，或者 PYTHON=<解释器>
py="${PYTHON:-}"
if [[ -z "${py}" ]]; then
  py=python3
  [[ -x "${repo_root}/.venv/bin/python" ]] && py="${repo_root}/.venv/bin/python"
fi
{
  echo "== ${name}: 目标 (${goal_x}, ${goal_y}) $([[ ${sent} -eq 1 ]] && echo 已发 || echo 没发出去)；LOC_MAP_PATH=${LOC_MAP_PATH:-} LOC_INS_ONLY=${LOC_INS_ONLY:-0} MAP_SWITCH=${MAP_SWITCH:-}"
  "${py}" "${nav_root}/planning/tools/check_planning_mcap.py" "${out}" \
    --layout "${nav_root}/planning/script/planning_io.json" || true
  if [[ -f "${out}/logs/planning.INFO" ]]; then
    p="${out}/logs/planning.INFO"
    echo "mission_state: $(grep -o 'mission_state: [a-z_]*' "${p}" | awk '{print $2}' | sort | uniq -c | awk '{printf "%s %s, ", $2, $1}')"
    echo "CMD: $(grep -c 'CMD: ' "${p}" || true) 次；头两条："
    grep -m 2 'CMD: ' "${p}" | sed 's/^/  /' || true
    for k in "Start adjusted" "Goal adjusted" "Start point is in collision" "Initial pose is in collision" \
             "No free point found near start" "Map switch" "crosswalks in" "OSM file not exist" "Failed to parse OSM"; do
      echo "${k}: $(grep -c "${k}" "${p}" || true)"
    done
  else
    echo "没找到这次规划的 glog（/tmp/planning.*.log.INFO.*）"
  fi
  # 给人看的版本：摊平到地图平面、换成 bag 时间、代价地图方向翻正；地图用规划这次用的那张（切过图就是切的那张），
  # 人行横道从同一个地图目录的 semantic.osm 读
  viz_args=()
  [[ -n "${MAP_SWITCH:-}" ]] && viz_args+=(--map "${MAP_SWITCH}")
  osm_dir="${MAP_SWITCH:-${LOC_MAP_PATH:-}}"
  [[ -n "${osm_dir}" ]] && ls "${osm_dir}"/2d/*/semantic.osm > /dev/null 2>&1 && viz_args+=(--osm "${osm_dir}")
  echo "== 可视化（tools/planning_mcap_viz.py）"
  "${py}" "${nav_root}/planning/tools/planning_mcap_viz.py" "${out}" "${viz_args[@]}" || echo "可视化没生成"
} 2>&1 | tee "${out}/report.txt"
