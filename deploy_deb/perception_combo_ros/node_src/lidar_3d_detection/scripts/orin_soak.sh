#!/usr/bin/env bash
# Long replay on the Orin (README.md): the installed node (the .deb, or a colcon install) on a bag played in a
# loop at its own pace, its objects recorded, its memory and tegrastats logged; the second half under load (the
# same bag replayed offline on the GPU in a loop, and 4 busy CPU cores).
#
#   orin_soak.sh BAG CONFIG MODEL_DIR MINUTES OUT       CONFIG / MODEL_DIR "-": the installed ones
#   orin_soak.sh bags/rosbag2_0921_600 - - 30 data/soak
#
# LIDAR3D_INSTALL: the install (default /opt/seres/perception, else this workspace's install/).
# Writes <out>/node.log, live/ (the recorded objects), rss.log (epoch s, kB), tegrastats.log, load.log (when the
# load started and stopped), play.log, record.log. The algorithm repository's tools/deploy/objects_bag.py checks
# live/ pass by pass, tools/deploy/soak_report.py sums up the rest. ROS_DOMAIN_ID defaults to 77.
set -u
if [ $# -ne 5 ]; then
  sed -n '2,13p' "$0" >&2
  exit 2
fi
BAG=$1 CONFIG=$2 MODEL=$3 MINUTES=$4 OUT=$5
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
INSTALL=${LIDAR3D_INSTALL:-/opt/seres/perception}
[ -d "$INSTALL/lidar_3d_detection" ] || INSTALL=$(cd "$HERE/../../.." && pwd)/install
PKG=$INSTALL/lidar_3d_detection
[ "$CONFIG" = - ] && CONFIG=$PKG/config/lidar_3d_detection.yaml
[ "$MODEL" = - ] && MODEL=$PKG/model
OBJECTS=$(sed -n 's/^  objects_topic: "\(.*\)"/\1/p' "$CONFIG")
OBJECTS=${OBJECTS:-/perception/lidar_3d/objects}
# the node's executable, not this script or the bag replay (whose paths also have lidar_3d_detection in them);
# ps -C cannot match it: process names are cut to 15 characters
NODE_RE='lib/lidar_3d_detection/lidar_3d_detection( |$)'
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
set +u
source /opt/ros/humble/setup.bash
source "$INSTALL/local_setup.bash"
set -u
if [ -e "$OUT/live" ]; then
  echo "$OUT/live exists" >&2
  exit 1
fi
mkdir -p "$OUT"
half=$((MINUTES * 30))

LIDAR3D_CONFIG=$CONFIG LIDAR3D_MODEL_DIR=$MODEL "$PKG/bin/start_lidar_3d_detection.sh" > "$OUT/node.log" 2>&1 &
for _ in $(seq 180); do
  grep -q "lidar_3d_detection up" "$OUT/node.log" && break
  sleep 1
done
if ! grep -q "lidar_3d_detection up" "$OUT/node.log"; then
  echo "the node did not come up in 180 s" >&2
  pkill -INT -f "$NODE_RE"
  exit 1
fi
ros2 bag record -o "$OUT/live" "$OBJECTS" > "$OUT/record.log" 2>&1 &
tegrastats --interval 5000 --logfile "$OUT/tegrastats.log" > /dev/null 2>&1 &
tegra=$!
(while sleep 30; do echo "$(date +%s) $(ps -o rss= -p "$(pgrep -f "$NODE_RE" | head -1)" 2> /dev/null)"; done) \
  > "$OUT/rss.log" &
rss=$!
sleep 3
# --delay: let DDS match the player's publishers with the node first (T29: the first frame was lost without it)
ros2 bag play --loop --delay 3 "$BAG" > "$OUT/play.log" 2>&1 &

sleep $half
echo "load on $(date +%s)" > "$OUT/load.log"
setsid bash -c "
  for i in 1 2 3 4; do (while :; do :; done) & done
  while :; do
    $PKG/lib/lidar_3d_detection/lidar_3d_bag_replay --bag $BAG --config $CONFIG --model-dir $MODEL --precision fp16 \
        --gpu --threads 4 --out /tmp/soak_load.npz 2>&1 | tail -1
  done" >> "$OUT/load.log" 2>&1 &
load=$!
sleep $half
kill -TERM -- -$load 2> /dev/null
pkill -f soak_load.npz
echo "load off $(date +%s)" >> "$OUT/load.log"

# SIGINT as Ctrl-C (rclcpp and rosbag2 handle it; record writes its metadata), then up to 10 s. pkill, not
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
kill $rss $tegra 2> /dev/null
stop "$NODE_RE"
sleep 2
rm -f /tmp/soak_load.npz
ros2 bag info "$OUT/live"
