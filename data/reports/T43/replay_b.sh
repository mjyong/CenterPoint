#!/bin/bash
# T43 B only. Publish the goal after planning has created Tracking.
cd /home/engineers/junyongmou/robot/ultralytics/d20car_predict/CenterPoint
export ROS_DOMAIN_ID=42
export CONDA_BUILD=0
prefix=/public/zjst/robot/planning_humble_env
export MAMBA_ROOT_PREFIX="${prefix}/root"
set +u
eval "$("${prefix}/micromamba" shell hook -s bash)"
micromamba activate "${prefix}/env"
# shellcheck disable=SC1091
source "${prefix}/colcon/install/setup.bash"
set -u

name=ins
out="$PWD/work_dirs/planning_io_t43_${name}"
rm -rf "$out"
start=$(date +%s)
ros2 bag record -s mcap -o "$out" \
  /lightning/odom /map/occupancy_grid /tf /tf_static \
  /planning/debug/cost_map_production/occmap \
  /planning/debug/goal_pose /planning/debug/path_tracking \
  /goal_pose /cmd_vel \
  > "data/reports/T43/record_${name}.txt" 2>&1 &
rec=$!
env LOC_INS_ONLY=1 bash Nav_D20/planning/script/run_planning_stack_debug.sh --once \
  > "data/reports/T43/run_${name}.txt" 2>&1 &
stack=$!

ok=0
for i in $(seq 1 90); do
  latest=$(find /tmp -name 'planning.gpu1.junyongmou.log.INFO.*' -newermt "@${start}" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -n "${latest}" ]] && grep -q "Create mission: Tracking" "${latest}"; then
    sleep 3
    ros2 service call /planning/debug/set_pub_flag std_srvs/srv/SetBool "{data: true}" >> "data/reports/T43/run_${name}.txt" 2>&1 || true
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: map}, pose: {position: {x: 80.0, y: 126.9, z: 0.0}, orientation: {w: 1.0}}}" \
      >> "data/reports/T43/run_${name}.txt" 2>&1 || true
    echo "goal sent after planning up try ${i} log ${latest}"
    ok=1
    break
  fi
  sleep 1
done
echo "enabled=${ok}"

for i in $(seq 1 200); do
  if ! pgrep -f "ros2 bag play" >/dev/null; then
    echo "bag play gone at ${i}"
    break
  fi
  sleep 2
done
sleep 2
# The debug script waits on every child. Stop them so it returns.
pkill -P "${stack}" 2>/dev/null || true
sleep 1
kill "${stack}" 2>/dev/null || true
wait "${stack}" 2>/dev/null || true
sleep 1
kill -INT "${rec}" 2>/dev/null || true
wait "${rec}" 2>/dev/null || true
cp -a /tmp/nav_stack_debug/loc.log "data/reports/T43/loc_${name}.log" || echo "no loc log"
echo DONE_B
