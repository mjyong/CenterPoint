#!/bin/bash
# Run a command in the nav-dev container (Dockerfile.nav-dev): the repository at /repo (the working directory),
# /public for bags, maps and models, one GPU, and the container's own network, so its ROS graph meets nobody else's
# (not another replay on this server, not the dog). NAV_STACK_ROOT points the replay scripts at the prefixes
# build_stack.sh made (<repo>/build/nav_dev/install), so they run the stack from there instead of micromamba.
#
#   bash Nav_D20/docker/run.sh <command...>
#   e.g. bash Nav_D20/docker/run.sh bash Nav_D20/docker/build_stack.sh
#        NAV_GPU=5 bash Nav_D20/docker/run.sh bash Nav_D20/planning/script/replay_planning_check.sh t58 78.7 226.6
#   NAV_GPU (default 0), NAV_IMAGE (default nav-dev), NAV_PUBLIC_RW=1 mounts /public writable (default read-only)
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)
exec docker run --rm "${tty[@]}" --gpus "\"device=${NAV_GPU:-0}\"" --shm-size 8g \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${repo}:/repo" -w /repo \
  -v "/public:/public:$([[ -n "${NAV_PUBLIC_RW:-}" ]] && echo rw || echo ro)" \
  -e NAV_STACK_ROOT=/repo/build/nav_dev/install \
  ${LOC_MAP_PATH:+-e LOC_MAP_PATH="${LOC_MAP_PATH}"} ${MAP_SWITCH:+-e MAP_SWITCH="${MAP_SWITCH}"} \
  ${PLANNING_STARTUP_MAP:+-e PLANNING_STARTUP_MAP="${PLANNING_STARTUP_MAP}"} \
  ${LOC_INS_YAW_OFFSET:+-e LOC_INS_YAW_OFFSET="${LOC_INS_YAW_OFFSET}"} ${LOC_INS_ONLY:+-e LOC_INS_ONLY="${LOC_INS_ONLY}"} \
  ${LIVE_PERCEPTION_MODEL:+-e LIVE_PERCEPTION_MODEL="${LIVE_PERCEPTION_MODEL}"} \
  ${BAG_PLAY_EXTRA:+-e BAG_PLAY_EXTRA="${BAG_PLAY_EXTRA}"} ${RECORD_CLOUD:+-e RECORD_CLOUD="${RECORD_CLOUD}"} \
  "${NAV_IMAGE:-nav-dev}" "$@"
