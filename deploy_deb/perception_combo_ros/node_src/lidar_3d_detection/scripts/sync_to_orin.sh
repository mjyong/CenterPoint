#!/usr/bin/env bash
# Copy this deployment tree (deploy_deb: perception_combo_ros and slam/interface) to the Orin, to build the
# .deb there (scripts/build_deb.sh). The model with its engines is built and kept on the Orin itself.
#
#   node_src/lidar_3d_detection/scripts/sync_to_orin.sh                 # to orin:~/seres_ws
#   node_src/lidar_3d_detection/scripts/sync_to_orin.sh --bag DIR       # + a rosbag2 to ~/seres_ws/bags
#
# ORIN_HOST (default orin, ~/.ssh/config) and ORIN_DIR (default ~/seres_ws) choose the target. Build outputs
# (build/, install/, log/, build_deb/, *.deb) on either side are left alone.
set -euo pipefail
HOST=${ORIN_HOST:-orin}
DIR=${ORIN_DIR:-seres_ws}
TREE=$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)     # deploy_deb
BAGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --bag) BAGS+=("$2"); shift 2 ;;
    *) sed -n '2,9p' "$0"; exit 2 ;;
  esac
done
ssh "$HOST" "mkdir -p $DIR/bags"
rsync -az --delete --exclude 'build/' --exclude 'install/' --exclude 'log/' --exclude 'build_deb/' --exclude '*.deb' \
  --exclude '__pycache__' "$TREE/perception_combo_ros" "$TREE/slam" "$HOST:$DIR/"
for b in "${BAGS[@]}"; do
  rsync -az "${b%/}" "$HOST:$DIR/bags/"
done
ssh "$HOST" "df -h / | tail -1; du -sh $DIR 2> /dev/null | cut -f1"
