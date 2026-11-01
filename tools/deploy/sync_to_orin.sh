#!/usr/bin/env bash
# Copy to the Orin what it needs, over the reverse tunnel (docs/DEPLOY_ORIN.md 2): the code, one exported
# model directory (ONNX + metadata; engines are built on the Orin itself) and a few scenes' frames.
# The Orin's root partition has only a few GB free: copy small subsets and delete them after use.
#
#   tools/deploy/sync_to_orin.sh                                  # code only
#   tools/deploy/sync_to_orin.sh --model data/deploy/nusc_pp      # ONNX + metadata, no engines
#   tools/deploy/sync_to_orin.sh --model data/deploy/nusc_pp_dyn --engines ubuntu
#       + the engines one machine built (<model>/machines/<host>, build_engines.py files them by hostname; the
#       Orin is "ubuntu"), only when they match the target's arch and L4T (build_engines.check_target)
#   tools/deploy/sync_to_orin.sh --frames data/runs/0921_164127_s000/frames --stride 3 --limit 100
#   tools/deploy/sync_to_orin.sh --dumps data/deploy/dumps        # reference detections (compare_backends --dump)
#   tools/deploy/sync_to_orin.sh --seq data/seq/0921_164127_s000 --limit 150   # a raw recording (lidar_3d_replay)
#   tools/deploy/sync_to_orin.sh --bag samples/rosbag2_2026_09_21-16_41_27_moving  # a rosbag2 (lidar_3d_bag_replay)
# ORIN_HOST (default orin, ~/.ssh/config) and ORIN_DIR (default ~/dogperc) choose the target.
set -euo pipefail
export LC_ALL=C          # list frames in the order Python's sorted() uses (compare_backends.py --stride / --limit)
HOST=${ORIN_HOST:-orin}
DIR=${ORIN_DIR:-dogperc}
cd "$(dirname "$0")/../.."
MODEL="" ; ENGINES="" ; DUMPS="" ; FRAMES=() ; SEQS=() ; BAGS=() ; STRIDE=1 ; LIMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --engines) ENGINES=$2; shift 2 ;;
    --frames) FRAMES+=("$2"); shift 2 ;;
    --seq) SEQS+=("$2"); shift 2 ;;
    --bag) BAGS+=("$2"); shift 2 ;;
    --stride) STRIDE=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --dumps) DUMPS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

# rsync creates only the last directory of a target: make the scene directories first
dirs="$DIR/repo $DIR/models $DIR/data"
for f in "${FRAMES[@]}"; do dirs="$dirs $DIR/data/$(basename "$(dirname "$f")")/frames"; done
for s in "${SEQS[@]}"; do dirs="$dirs $DIR/seq/$(basename "$s")/scans"; done
for b in "${BAGS[@]}"; do dirs="$dirs $DIR/bags"; done
ssh "$HOST" "mkdir -p $dirs"
# build*/: the build directories (build/lidar3d, deploy_deb/.../build_deb) are per machine; excluded ones survive --delete
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' --exclude 'build*/' --exclude 'install/' --exclude 'log/' \
  dog_perception tools deploy_deb configs tests kygi665_msgs requirements_dog.txt "$HOST:$DIR/repo/"
# the board's copy has no .git: deploy_deb's build_deb.sh versions the .deb from this ("count hash [dirty]")
if commit=$(git rev-list --count HEAD 2> /dev/null); then
  commit="$commit $(git rev-parse --short HEAD)"
  git diff --quiet HEAD -- deploy_deb || commit="$commit dirty"
  ssh "$HOST" "echo '$commit' > $DIR/repo/deploy_deb/GIT_COMMIT"
fi
if [ -n "$MODEL" ]; then
  # engines.json is the build log of the engines next to it (build_engines.py): the Orin keeps its own
  rsync -az --exclude 'engines.json' --exclude 'machines/' --include '*.onnx' --include '*.json' --exclude '*' \
    "$MODEL/" "$HOST:$DIR/models/$(basename "$MODEL")/"
fi
if [ -n "$ENGINES" ]; then
  [ -n "$MODEL" ] || { echo "--engines needs --model" >&2; exit 2; }
  src="$MODEL/machines/$ENGINES"
  [ -d "$src" ] || { echo "no engines for $ENGINES ($src)" >&2; exit 1; }
  remote_arch=$(ssh "$HOST" uname -m)
  remote_l4t=$(ssh "$HOST" "cat /etc/nv_tegra_release 2>/dev/null || true")
  python3 - "$src/engines.json" "$remote_arch" "$remote_l4t" << 'PY'
import json, sys
sys.path.insert(0, "tools/deploy")
from build_engines import check_target
check_target(json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3])
PY
  rsync -az "$src/" "$HOST:$DIR/models/$(basename "$MODEL")/"
  echo "engines: $ENGINES -> $HOST:$DIR/models/$(basename "$MODEL")/"
fi
for f in "${FRAMES[@]}"; do
  scene=$(basename "$(dirname "$f")")
  list=$(mktemp)
  ls "$f" | grep '\.npy$' | awk -v s="$STRIDE" 'NR % s == 1 || s == 1' > "$list"
  [ "$LIMIT" -gt 0 ] && head -n "$LIMIT" "$list" > "$list.cut" && mv "$list.cut" "$list"
  rsync -az --files-from="$list" "$f/" "$HOST:$DIR/data/$scene/frames/"
  echo "$scene: $(wc -l < "$list") frames"
  rm -f "$list"
done
# raw recordings: meta, IMU, odometry and the first --limit scans (all with --limit 0)
for s in "${SEQS[@]}"; do
  name=$(basename "$s")
  rsync -az "$s/meta.json" "$HOST:$DIR/seq/$name/"
  for f in imu.npy odom.npy; do [ -f "$s/$f" ] && rsync -az "$s/$f" "$HOST:$DIR/seq/$name/"; done
  list=$(mktemp)
  ls "$s/scans" | grep '\.npz$' > "$list"
  [ "$LIMIT" -gt 0 ] && head -n "$LIMIT" "$list" > "$list.cut" && mv "$list.cut" "$list"
  rsync -az --files-from="$list" "$s/scans/" "$HOST:$DIR/seq/$name/scans/"
  echo "$name: $(wc -l < "$list") scans"
  rm -f "$list"
done
if [ -n "$DUMPS" ]; then
  rsync -az "$DUMPS/" "$HOST:$DIR/data/dumps/"
fi
ssh "$HOST" "df -h / | tail -1; du -sh $DIR 2>/dev/null"
# rosbag2 directories (metadata.yaml + .db3), whole
for b in "${BAGS[@]}"; do
  rsync -az "${b%/}" "$HOST:$DIR/bags/"
  echo "$(basename "${b%/}"): $(du -sh "$b" | cut -f1)"
done
