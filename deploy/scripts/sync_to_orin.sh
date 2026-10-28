#!/usr/bin/env bash
# Copy to the Orin what it needs, over the reverse tunnel (docs/DEPLOY_ORIN.md 2): the code, one exported
# model directory (ONNX + metadata; engines are built on the Orin itself) and a few scenes' frames.
# The Orin's root partition has only a few GB free: copy small subsets and delete them after use.
#
#   deploy/scripts/sync_to_orin.sh                                  # code only
#   deploy/scripts/sync_to_orin.sh --model data/deploy/nusc_pp      # + the model directory
#   deploy/scripts/sync_to_orin.sh --frames data/runs/0921_164127_s000/frames --stride 3 --limit 100
#   deploy/scripts/sync_to_orin.sh --dumps data/deploy/dumps        # reference detections (compare_backends --dump)
# ORIN_HOST (default orin, ~/.ssh/config) and ORIN_DIR (default ~/dogperc) choose the target.
set -euo pipefail
export LC_ALL=C          # list frames in the order Python's sorted() uses (compare_backends.py --stride / --limit)
HOST=${ORIN_HOST:-orin}
DIR=${ORIN_DIR:-dogperc}
cd "$(dirname "$0")/../.."
MODEL="" ; DUMPS="" ; FRAMES=() ; STRIDE=1 ; LIMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --frames) FRAMES+=("$2"); shift 2 ;;
    --stride) STRIDE=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --dumps) DUMPS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

ssh "$HOST" "mkdir -p $DIR/repo $DIR/models $DIR/data"
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
if [ -n "$DUMPS" ]; then
  rsync -az "$DUMPS/" "$HOST:$DIR/data/dumps/"
fi
ssh "$HOST" "df -h / | tail -1; du -sh $DIR 2>/dev/null"
