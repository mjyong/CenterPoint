#!/usr/bin/env bash
# Long replay on the Orin (T30, DEPLOY_ORIN.md 8): the node on a bag played in a loop at its own pace, ~/objects
# recorded, the node's memory and tegrastats logged; the second half under load (the same bag replayed offline on
# the GPU in a loop, and 4 busy CPU cores). Run on the board from ~/dogperc, after colcon build:
#
#   repo/deploy/scripts/orin_soak.sh bags/rosbag2_0921_600 data/kygi.yaml models/nusc_pp_dyn 30 data/soak
#
# Writes <out>/node.log, live/ (the recorded ~/objects), rss.log (epoch s, kB), tegrastats.log, load.log (when the
# load started and stopped), play.log, record.log. tools/deploy/objects_bag.py checks live/ pass by pass,
# tools/deploy/soak_report.py sums up the rest. ROS_DOMAIN_ID defaults to 77 (away from the robot's programs).
set -u
if [ $# -ne 5 ]; then
  echo "usage: $0 BAG CONFIG MODEL_DIR MINUTES OUT" >&2
  exit 2
fi
BAG=$1 CONFIG=$2 MODEL=$3 MINUTES=$4 OUT=$5
HERE=$(cd "$(dirname "$0")" && pwd)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
set +u
source /opt/ros/humble/setup.bash
source "$HERE/../ros2/install/setup.bash"
set -u
if [ -e "$OUT/live" ]; then
  echo "$OUT/live exists" >&2
  exit 1
fi
mkdir -p "$OUT"
half=$((MINUTES * 30))

ros2 launch dogperc_ros dogperc.launch.py config:="$CONFIG" model_dir:="$MODEL" > "$OUT/node.log" 2>&1 &
launch=$!
for _ in $(seq 180); do
  grep -q "dog perception up" "$OUT/node.log" && break
  sleep 1
done
if ! grep -q "dog perception up" "$OUT/node.log"; then
  echo "the node did not come up in 180 s" >&2
  pkill -INT -f dogperc_node
  exit 1
fi
ros2 bag record -o "$OUT/live" /dog_perception/objects > "$OUT/record.log" 2>&1 &
tegrastats --interval 5000 --logfile "$OUT/tegrastats.log" > /dev/null 2>&1 &
tegra=$!
(while sleep 30; do echo "$(date +%s) $(ps -o rss= -C dogperc_node | head -1)"; done) > "$OUT/rss.log" &
rss=$!
sleep 3
# --delay: let DDS match the player's publishers with the node first (T29: the first frame was lost without it)
ros2 bag play --loop --delay 3 "$BAG" > "$OUT/play.log" 2>&1 &

sleep $half
echo "load on $(date +%s)" > "$OUT/load.log"
setsid bash -c "
  for i in 1 2 3 4; do (while :; do :; done) & done
  while :; do
    ros2 run dogperc_ros dogperc_bag_replay --bag $BAG --config $CONFIG --model-dir $MODEL --precision fp16 --gpu \
        --threads 4 --out /tmp/soak_load.npz 2>&1 | tail -1
  done" >> "$OUT/load.log" 2>&1 &
load=$!
sleep $half
kill -TERM -- -$load 2>/dev/null
pkill -f soak_load.npz
echo "load off $(date +%s)" >> "$OUT/load.log"

# SIGINT as Ctrl-C (rclcpp, rosbag2 and launch handle it; record writes its metadata), then up to 10 s. pkill, not
# kill $pid: background jobs of a script start with SIGINT ignored, only the ROS handlers take it back (T29)
stop() {
  pkill -INT -f "$1"
  for _ in $(seq 20); do
    pgrep -f "$1" > /dev/null || return 0
    sleep 0.5
  done
  echo "still running 10 s after SIGINT: $1" >&2
  pkill -TERM -f "$1"
}
stop "ros2 bag play"
sleep 3
stop "ros2 bag record"
kill $rss $tegra 2>/dev/null
stop dogperc_node
wait $launch 2>/dev/null
rm -f /tmp/soak_load.npz
ros2 bag info "$OUT/live"
