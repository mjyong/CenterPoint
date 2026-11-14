#!/bin/bash
# Foxglove on dog 8 while our navigation runs (nav_cw.sh use-new): foxglove_bridge on the robot's domain, as root
# (the robot's shared memory is root's), with a topic whitelist (unrestricted, the bridge quits: iceoryx's publisher
# table fills up, T40) and only /goal_pose publishable from Foxglove (the 3D panel's "Publish pose" tool sends the
# goal; nothing else, so Foxglove can never command /cmd_vel).
#
#   sudo bash /opt/seres/nav_cw/nav_cw_foxglove.sh            then in Foxglove: ws://<dog>:8765,
#                                                             Layout -> Import Nav_D20/planning/script/planning_live.json
#   NAV_CW_FOXGLOVE_PORT (8765), NAV_CW_PREFIX (/opt/seres/nav_cw: its planning prefix gives the message definitions),
#   NAV_CW_FOXGLOVE_EXTRA="^/some/topic$ ..." more topics to let through
# The planner's debug topics need pub_flag (nav_cw.sh configure turns it on; NAV_CW_DEBUG=0 keeps the repo's off).
set -euo pipefail
prefix="${NAV_CW_PREFIX:-/opt/seres/nav_cw}"
port="${NAV_CW_FOXGLOVE_PORT:-8765}"
# what planning_live.json shows (tests/dog/test_nav_cw.py checks the two agree), the transforms, and the goal
TOPICS=(
  '^/planning/debug/(path_tracking|goal_pose|collision_checker|crosswalk|diagnostic)$'
  '^/planning/debug/(cost_map_production|global_map_manager)/occmap$'
  '^/planning/debug/plotjuggler/string$'
  '^/map/occupancy_grid$' '^/fused_point_cloud$' '^/lightning/odom$' '^/cmd_vel$'
  '^/perception/lidar_3d/objects$' '^/goal_pose$' '^/clicked_point$' '^/initialpose$' '^/tf$' '^/tf_static$'
)
read -r -a extra <<< "${NAV_CW_FOXGLOVE_EXTRA:-}"
TOPICS+=("${extra[@]}")
list=""
for t in "${TOPICS[@]}"; do list="${list:+${list}, }\"${t}\""; done
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
[[ -f "${prefix}/install/planning/setup.bash" ]] && source "${prefix}/install/planning/setup.bash"
set -u
# the robot's DDS (T34: CycloneDDS, its config, domain 0 unless set)
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///opt/ros/cyclonedds.xml}"
echo "foxglove_bridge on ${port}, domain ${ROS_DOMAIN_ID:-0}; topics: ${list}"
exec ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:="${port}" \
  -p "topic_whitelist:=[${list}]" \
  -p "client_topic_whitelist:=[\"^/goal_pose$\"]" \
  -p "capabilities:=[\"clientPublish\", \"connectionGraph\"]"
