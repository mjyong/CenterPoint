#!/bin/bash
# T43 A then B. Domain 42. Does not touch the dog.
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

one() {
  local name=$1
  shift
  local out="$PWD/work_dirs/planning_io_t43_${name}"
  rm -rf "$out"
  ros2 bag record -s mcap -o "$out" \
    /lightning/odom /map/occupancy_grid /tf /tf_static \
    /planning/debug/cost_map_production/occmap \
    /planning/debug/goal_pose /planning/debug/path_tracking \
    /goal_pose /cmd_vel \
    > "data/reports/T43/record_${name}.txt" 2>&1 &
  local rec=$!
  env "$@" bash Nav_D20/planning/script/run_planning_stack_debug.sh --once \
    > "data/reports/T43/run_${name}.txt" 2>&1 &
  local stack=$!
  local enabled=0
  local i
  for i in $(seq 1 50); do
    if ros2 service list 2>/dev/null | grep -q '/planning/debug/set_pub_flag'; then
      ros2 service call /planning/debug/set_pub_flag std_srvs/srv/SetBool "{data: true}" >> "data/reports/T43/run_${name}.txt" 2>&1 || true
      ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
        "{header: {frame_id: map}, pose: {position: {x: 80.0, y: 126.9, z: 0.0}, orientation: {w: 1.0}}}" \
        >> "data/reports/T43/run_${name}.txt" 2>&1 || true
      enabled=1
      echo "goal sent ${name} try ${i}"
      break
    fi
    sleep 1
  done
  echo "enabled=${enabled}"
  wait "$stack" || echo "stack ${name} exit $?"
  sleep 2
  kill -INT "$rec" 2>/dev/null || true
  wait "$rec" 2>/dev/null || true
  cp -a /tmp/nav_stack_debug/loc.log "data/reports/T43/loc_${name}.log" || echo "no loc log ${name}"
  find "$out" -type f -printf '%s %p\n' | head
}

echo "==== A map ===="
one map LOC_MAP_PATH="$PWD/data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13"
echo "==== B ins ===="
one ins LOC_INS_ONLY=1
echo DONE_REPLAY
