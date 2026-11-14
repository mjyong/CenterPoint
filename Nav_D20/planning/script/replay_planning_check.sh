#!/bin/bash
# 回放一份 bag 跑整套（run_planning_stack_debug.sh --once），录规划的输入输出，发一个目标点；
# bag 放完以后查录包（tools/check_planning_mcap.py），数规划日志里的关键行。T41–T43 的手工步骤收在这里（T46 复核）。
#
#   bash Nav_D20/planning/script/replay_planning_check.sh <名字> <目标 x> <目标 y> [--bag BAG]
#   例：LOC_MAP_PATH=data/loc_maps/ad68517c-... bash Nav_D20/planning/script/replay_planning_check.sh t47_map 80.0 126.9
#
# 环境变量同 run_planning_stack_debug.sh：LOC_MAP_PATH、LOC_INS_ONLY、LOC_INS_YAW_OFFSET（不设按 bag 日期定）、PLANNING_VIN；ROS_DOMAIN_ID 默认 42（会发 /cmd_vel，不要用狗的域）。
# MAP_SWITCH=<地图目录>（有 global_map.yaml 的）：发目标之前先发 /map/switch，规划换成这张图，读它的 semantic.osm。
# 输出在 work_dirs/planning_io_<名字>/：录包（mcap）、logs/（各进程日志、规划的 glog）、report.txt（查包和关键行）、
# viz/planning_viz.mcap 和 .png（tools/planning_mcap_viz.py：在 Foxglove 里用 script/planning_viz.json 看规划，
# script/loc_viz.json 看定位）。
# 点云这类大话题（定位的当前帧 /lightning/current_scan_world、地图 /lightning/map_cloud，RECORD_CLOUD=1 时还有感知的
# /fused_point_cloud）录到旁边的 /tmp/replay_<名字>_heavy，只给转换用，转完删掉（KEEP_HEAVY=1 留着）。
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
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# 环境：nav-dev 镜像里（NAV_STACK_ROOT）用 build_stack.sh 编的各模块前缀，否则 micromamba（nav_env.sh）
# shellcheck disable=SC1091
source "${nav_root}/planning/script/nav_env.sh"
nav_env_setup

rm -rf "${out}"
mkdir -p "$(dirname "${out}")"
stamp="$(date +%s)"
ros2 bag record -s mcap -o "${out}" \
  /lightning/odom /map/occupancy_grid /tf /tf_static \
  /planning/debug/cost_map_production/occmap /planning/debug/global_map_manager/occmap \
  /planning/debug/goal_pose /planning/debug/path_tracking /planning/debug/plotjuggler/string \
  /planning/debug/diagnostic /planning/debug/collision_checker /planning/debug/teb_planner /planning/debug/following \
  /goal_pose /cmd_vel /map/switch \
  /rte/rdap/rtk/inspvax /rhr/topic/event_info /rhr/topic/alarm_info /perception/lidar_3d/objects \
  > "/tmp/replay_${name}_record.txt" 2>&1 &
rec=$!
heavy="/tmp/replay_${name}_heavy"
rm -rf "${heavy}"
ros2 bag record -s mcap -o "${heavy}" /lightning/current_scan_world /lightning/map_cloud \
  $([[ "${RECORD_CLOUD:-0}" == 1 ]] && echo /fused_point_cloud) > "/tmp/replay_${name}_record_heavy.txt" 2>&1 &
rec_heavy=$!
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
kill -INT "${rec}" "${rec_heavy}" 2>/dev/null || true
wait "${rec}" 2>/dev/null || true
wait "${rec_heavy}" 2>/dev/null || true

mkdir -p "${out}/logs"
cp /tmp/nav_stack_debug/*.log "/tmp/replay_${name}_record.txt" "/tmp/replay_${name}_record_heavy.txt" \
   "/tmp/replay_${name}_stack.txt" "${out}/logs/" 2>/dev/null || true
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
# 不带 ROS / conda 的 PYTHONPATH：那里的 numpy 会盖过 .venv 的（T54 的可视化就是这么失败的）；工具只要仓库在路径上
py_clean() { env -u PYTHONPATH PYTHONPATH="${repo_root}" "${py}" "$@"; }
{
  echo "== ${name}: 目标 (${goal_x}, ${goal_y}) $([[ ${sent} -eq 1 ]] && echo 已发 || echo 没发出去)；LOC_MAP_PATH=${LOC_MAP_PATH:-} LOC_INS_ONLY=${LOC_INS_ONLY:-0} MAP_SWITCH=${MAP_SWITCH:-}"
  py_clean "${nav_root}/planning/tools/check_planning_mcap.py" "${out}" \
    --layout "${nav_root}/planning/script/planning_io.json" || true
  if [[ -f "${out}/logs/planning.INFO" ]]; then
    p="${out}/logs/planning.INFO"
    echo "mission_state: $(grep -o 'mission_state: [a-z_]*' "${p}" | awk '{print $2}' | sort | uniq -c | awk '{printf "%s %s, ", $2, $1}')"
    echo "CMD: $(grep -c 'CMD: ' "${p}" || true) 次；头两条："
    grep -m 2 'CMD: ' "${p}" | sed 's/^/  /' || true
    # 指令的中位：狗沿 bag 往前走时 vx 应该大多是正的（T48 狗头反了，vx 约 0、vy 横着）
    echo "CMD 中位 vx vy ω：$(for k in 2 1 0; do grep 'CMD: ' "${p}" | awk -v k=${k} '{print $(NF-k)}' | sort -g |
      awk '{a[NR]=$1} END {if (NR) printf "%.3f ", a[int((NR+1)/2)]}'; done)；vx > 0.05 的比例 $(grep 'CMD: ' "${p}" |
      awk '{n++; if ($(NF-2) > 0.05) k++} END {if (n) printf "%.0f%%", 100*k/n}')"
    echo "车型：$(grep -m 1 '规划车型' "/tmp/replay_${name}_stack.txt" || echo 没找到)；$(grep -m 1 -o 'Load robot config:.*' "${p}" || true)"
    echo "定位：$(grep -m 1 'INS 朝向偏置' "/tmp/replay_${name}_stack.txt" || echo 没找到)；$(grep -m 1 -o 'system.ins_yaw_offset = .*' /tmp/nav_stack_debug/loc.log 2> /dev/null || echo 定位日志里没有，lightning 没重新编译？)"
    for k in "Start adjusted" "Goal adjusted" "Start point is in collision" "Initial pose is in collision" \
             "No free point found near start" "Map switch" "startup map" "crosswalks in" "OSM file not exist" "Failed to parse OSM" \
             "Data timeout exceeded" "Data recovered, resuming" "local planner solver failed" "Crosswalk yield:"; do
      echo "${k}: $(grep -c "${k}" "${p}" || true)"
    done
    # 人行横道让行的每次状态切换（approach / align / wait / go / cross，括号里是门控的理由）
    grep "Tracking: crosswalk yield\|Crosswalk yield:" "${p}" | head -40 | sed 's/^[^]]*\] //' | sed 's/^/  /' || true
  else
    echo "没找到这次规划的 glog（/tmp/planning.*.log.INFO.*）"
  fi
  # 定位：LiDAR 回调卡住几次、处理本身慢几次，退出时打印的各步耗时（T49：回调隔几秒卡 0.9 s，odom 跟着跳）
  l="${out}/logs/loc.log"
  if [[ -f "${l}" ]]; then
    echo "定位：LiDAR 回调间隔过大 $(grep -c 'LiDAR 回调间隔过大' "${l}" || true) 次，ProcessLidar 超过 0.2 s $(grep -c 'ProcessLidar 用了' "${l}" || true) 次（最长 $(grep -o 'ProcessLidar 用了 [0-9.]*' "${l}" | awk '{print $3}' | sort -g | tail -1)），雷达断流 $(grep -c '检测到雷达断流' "${l}" || true) 次"
    sed -n '/Printing run time =====/,/Printing run time end/p' "${l}" | sed 's/^[^]]*\] //' | sed 's/^/  /'
  fi
  # 给人看的版本：摊平到地图平面、换成 bag 时间、代价地图方向翻正；地图用规划这次用的那张（切过图就是切的那张），
  # 人行横道从同一个地图目录的 semantic.osm 读
  viz_args=()
  [[ -n "${MAP_SWITCH:-}" ]] && viz_args+=(--map "${MAP_SWITCH}")
  osm_dir="${MAP_SWITCH:-${LOC_MAP_PATH:-}}"
  [[ -n "${osm_dir}" ]] && ls "${osm_dir}"/2d/*/semantic.osm > /dev/null 2>&1 && viz_args+=(--osm "${osm_dir}")
  # 定位的地图点云：定位用的那张（LOC_MAP_PATH）；没给就用录到的 /lightning/map_cloud
  [[ -n "${LOC_MAP_PATH:-}" && -f "${LOC_MAP_PATH}/3d/global.pcd" ]] && viz_args+=(--loc-map "${LOC_MAP_PATH}")
  heavy_in=()
  ls "${heavy}"/*.mcap > /dev/null 2>&1 && heavy_in=("${heavy}")
  echo "== 可视化（tools/planning_mcap_viz.py）"
  py_clean "${nav_root}/planning/tools/planning_mcap_viz.py" "${out}" "${heavy_in[@]}" "${viz_args[@]}" \
    || echo "可视化没生成"
} 2>&1 | tee "${out}/report.txt"
[[ "${KEEP_HEAVY:-0}" == 1 ]] || rm -rf "${heavy}"
