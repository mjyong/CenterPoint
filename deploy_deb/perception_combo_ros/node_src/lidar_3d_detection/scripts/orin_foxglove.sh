#!/usr/bin/env bash
# Live view of the board in Foxglove (README.md): the installed node (with TF and the detector's input cloud for
# the viewer) on a bag played in a loop, and foxglove_bridge (WebSocket). A running lidar-3d-detection.service is
# stopped for the view (sudo -n) and started again at the end. On the PC wired to the board: Foxglove -> Open connection -> Foxglove
# WebSocket -> ws://192.168.123.100:8765, then Layout -> Import foxglove/lidar_3d_detection.json.
#
#   orin_foxglove.sh BAG [RATE]          run on the board (ssh from the PC); Ctrl-C stops everything
#
# ROS_DOMAIN_ID (default 77, away from the robot's programs), FOXGLOVE_PORT (8765), LIDAR3D_INSTALL (default
# /opt/seres/perception, else this workspace's install/). Logs in /tmp/lidar3d_foxglove.
# Needs foxglove_bridge: sudo apt install ros-humble-foxglove-bridge (README.md for a board without internet).
set -u
if [ $# -lt 1 ]; then
  sed -n '2,10p' "$0" >&2
  exit 2
fi
BAG=$1 RATE=${2:-1.0}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
INSTALL=${LIDAR3D_INSTALL:-/opt/seres/perception}
[ -d "$INSTALL/lidar_3d_detection" ] || INSTALL=$(cd "$HERE/../../.." && pwd)/install
PKG=$INSTALL/lidar_3d_detection
PORT=${FOXGLOVE_PORT:-8765}
LOG=/tmp/lidar3d_foxglove
NODE_RE='lib/lidar_3d_detection/lidar_3d_detection( |$)'
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
set +u
source /opt/ros/humble/setup.bash
source "$INSTALL/local_setup.bash"
set -u
if ! ros2 pkg prefix foxglove_bridge > /dev/null 2>&1; then
  echo "foxglove_bridge is not installed: sudo apt install ros-humble-foxglove-bridge" >&2
  exit 1
fi
# the .deb enables the service: it runs the same node (on the robot's ROS_DOMAIN_ID)
service=0
if systemctl is-active --quiet lidar-3d-detection 2> /dev/null; then
  if ! sudo -n systemctl stop lidar-3d-detection; then
    echo "lidar-3d-detection.service is running: sudo systemctl stop lidar-3d-detection, then run this again" >&2
    exit 1
  fi
  service=1
fi
if pgrep -f "$NODE_RE" > /dev/null; then
  echo "lidar_3d_detection is already running (another orin_foxglove.sh, orin_soak.sh or a manual start)" >&2
  exit 1
fi
mkdir -p "$LOG"

cleanup() {
  trap - EXIT INT TERM
  for p in "ros2 bag play" "$NODE_RE" "lib/foxglove_bridge/foxglove_bridge"; do pkill -INT -f "$p"; done
  sleep 2
  for p in "ros2 bag play" "$NODE_RE" "lib/foxglove_bridge/foxglove_bridge"; do pkill -TERM -f "$p"; done
  if [ $service = 1 ]; then sudo -n systemctl start lidar-3d-detection && echo "lidar-3d-detection.service started again"; fi
  echo "stopped (logs in $LOG)"
}
trap cleanup EXIT INT TERM

# the layout shows the detector's input cloud (/perception/lidar_3d/cloud, 5 sweeps, a few MB a frame)
"$PKG/bin/start_lidar_3d_detection.sh" -p publish_tf:=true -p publish_cloud:=true > "$LOG/node.log" 2>&1 &
for _ in $(seq 120); do
  grep -q "lidar_3d_detection up" "$LOG/node.log" && break
  sleep 1
done
if ! grep -q "lidar_3d_detection up" "$LOG/node.log"; then
  echo "the node did not come up in 120 s:" >&2
  tail -5 "$LOG/node.log" >&2
  exit 1
fi
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:="$PORT" > "$LOG/bridge.log" 2>&1 &
sleep 2
echo "node up, foxglove_bridge on port $PORT; in Foxglove open ws://<board>:$PORT, the board's addresses:"
hostname -I
echo "playing $BAG in a loop at rate $RATE (Ctrl-C, or pkill -INT -f orin_foxglove.sh, stops)"
# in the background: a signal interrupts `wait` at once (a foreground command would delay the trap until it ends)
ros2 bag play --loop --rate "$RATE" "$BAG" > "$LOG/play.log" 2>&1 &
wait $!
