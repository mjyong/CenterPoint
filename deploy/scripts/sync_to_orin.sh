#!/usr/bin/env bash
# Copy to the Orin what it needs, over the reverse tunnel (docs/DEPLOY_ORIN.md 2): the code, one exported
# model directory (ONNX + metadata; engines are built on the Orin itself) and a few scenes' frames.
# The Orin's root partition has only a few GB free: copy small subsets and delete them after use.
#
#   deploy/scripts/sync_to_orin.sh                                  # code only
#   deploy/scripts/sync_to_orin.sh --model data/deploy/nusc_pp      # + the model directory
#   deploy/scripts/sync_to_orin.sh --frames data/runs/0921_164127_s000/frames --stride 3 --limit 100
#   deploy/scripts/sync_to_orin.sh --dumps data/deploy/dumps        # reference detections (compare_backends --dump)
#   deploy/scripts/sync_to_orin.sh --seq data/seq/0921_164127_s000 --limit 150   # a raw recording (dogperc_replay)
# ORIN_HOST (default orin, ~/.ssh/config) and ORIN_DIR (default ~/dogperc) choose the target.
set -euo pipefail
export LC_ALL=C          # list frames in the order Python's sorted() uses (compare_backends.py --stride / --limit)
HOST=${ORIN_HOST:-orin}
DIR=${ORIN_DIR:-dogperc}
cd "$(dirname "$0")/../.."
MODEL="" ; DUMPS="" ; FRAMES=() ; SEQS=() ; STRIDE=1 ; LIMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --frames) FRAMES+=("$2"); shift 2 ;;
    --seq) SEQS+=("$2"); shift 2 ;;
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
ssh "$HOST" "mkdir -p $dirs"
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' --exclude 'build/' --exclude 'install/' --exclude 'log/' \
  dog_perception tools deploy configs tests kygi665_msgs requirements_dog.txt "$HOST:$DIR/repo/"
if [ -n "$MODEL" ]; then
  rsync -az --include '*.onnx' --include '*.json' --exclude '*' "$MODEL/" "$HOST:$DIR/models/$(basename "$MODEL")/"
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
