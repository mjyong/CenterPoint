#!/usr/bin/env bash
# Start the node from this install (the .deb's /opt/seres/perception, or a colcon install): source ROS 2 and the
# install, then run the node with the installed config and model. lidar-3d-detection.service runs this.
#
#   <install>/lidar_3d_detection/bin/start_lidar_3d_detection.sh [-p name:=value ...]   extra node parameters
#
# Environment (for the service: /etc/default/lidar-3d-detection):
#   ROS_DOMAIN_ID, RMW_IMPLEMENTATION    as for any ROS 2 node
#   LIDAR3D_CONFIG      default <install>/lidar_3d_detection/config/lidar_3d_detection.yaml
#   LIDAR3D_MODEL_DIR   default node.model_dir of the config (<install>/lidar_3d_detection/model)
#   LIDAR3D_THREADS     threads of the per-point preprocessing, default 4 (the Orin, T28B)
#   LIDAR3D_LOG_DIR     glog files, default /tmp/lidar_3d_detection_logs
#   ROS_SETUP           default /opt/ros/humble/setup.bash
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)    # <install>/lidar_3d_detection/bin
PKG=$(dirname "$HERE")
ROOT=$(dirname "$PKG")
ROS_SETUP=${ROS_SETUP:-/opt/ros/humble/setup.bash}
for f in "$ROS_SETUP" "$ROOT/local_setup.bash" "$PKG/lib/lidar_3d_detection/lidar_3d_detection"; do
  if [ ! -e "$f" ]; then
    echo "start_lidar_3d_detection.sh: $f not found" >&2
    exit 1
  fi
done
source "$ROS_SETUP"
source "$ROOT/local_setup.bash"
export SERES_PERCEPTION_INSTALL_DIR=$ROOT
args=(--ros-args -p "threads:=${LIDAR3D_THREADS:-4}")
if [ -n "${LIDAR3D_CONFIG:-}" ]; then args+=(-p "config:=$LIDAR3D_CONFIG"); fi
if [ -n "${LIDAR3D_MODEL_DIR:-}" ]; then args+=(-p "model_dir:=$LIDAR3D_MODEL_DIR"); fi
exec "$PKG/lib/lidar_3d_detection/lidar_3d_detection" "${args[@]}" "$@"
