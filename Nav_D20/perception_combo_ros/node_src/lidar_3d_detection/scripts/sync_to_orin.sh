#!/usr/bin/env bash
# Copy what the .deb is built from (Nav_D20/perception_combo_ros: common_interface and node_src/lidar_3d_detection;
# our node is part of the robot team's perception since 2026-11-14) to the Orin, to build the .deb there
# (scripts/build_deb.sh). The model with its engines is built and kept on the Orin itself.
#
#   node_src/lidar_3d_detection/scripts/sync_to_orin.sh                 # to orin:~/seres_ws
#   node_src/lidar_3d_detection/scripts/sync_to_orin.sh --bag DIR       # + a rosbag2 to ~/seres_ws/bags
#
# ORIN_HOST (default orin, ~/.ssh/config) and ORIN_DIR (default ~/seres_ws) choose the target. Build outputs
# (build/, install/, log/, build_deb/, *.deb) on either side are left alone.
set -euo pipefail
HOST=${ORIN_HOST:-orin}
DIR=${ORIN_DIR:-seres_ws}
TREE=$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)     # Nav_D20
BAGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --bag) BAGS+=("$2"); shift 2 ;;
    *) sed -n '2,9p' "$0"; exit 2 ;;
  esac
done
ssh "$HOST" "mkdir -p $DIR/bags $DIR/perception_combo_ros/node_src"
ex=(--exclude 'build/' --exclude 'install/' --exclude 'log/' --exclude 'build_deb/' --exclude '*.deb' --exclude '__pycache__')
rsync -az --delete "${ex[@]}" "$TREE/perception_combo_ros/common_interface" "$HOST:$DIR/perception_combo_ros/"
rsync -az --delete "${ex[@]}" "$TREE/perception_combo_ros/node_src/lidar_3d_detection" "$HOST:$DIR/perception_combo_ros/node_src/"
# the copy has no .git: build_deb.sh versions the .deb from this ("count hash [dirty]")
if commit=$(git -C "$TREE" rev-list --count HEAD 2> /dev/null); then
  commit="$commit $(git -C "$TREE" rev-parse --short HEAD)"
  git -C "$TREE" diff --quiet HEAD -- . || commit="$commit dirty"
  ssh "$HOST" "echo '$commit' > $DIR/GIT_COMMIT"
fi
for b in "${BAGS[@]}"; do
  rsync -az "${b%/}" "$HOST:$DIR/bags/"
done
ssh "$HOST" "df -h / | tail -1; du -sh $DIR 2> /dev/null | cut -f1"
