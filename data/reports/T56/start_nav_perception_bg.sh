#!/bin/bash
# cw-nav-perception. Our lidar_seg, grid_map_memory_node and lidar_3d_detection.
# ptz_data_detection_2d and perception_recharge stay the robot team's binaries.
set -u
OUR=/opt/seres/nav_cw/install/nav_perception
THEIRS=/opt/seres/nav_perception
source /opt/ros/humble/setup.bash
source "${OUR}/setup.bash"
export PERCEPTION_INSTALL_DIR="${OUR}"
export SERES_PERCEPTION_INSTALL_DIR="${OUR}"
export LD_LIBRARY_PATH="/opt/seres/nav_cw/deps/usr/lib/aarch64-linux-gnu:/opt/seres/nav_cw/deps/usr/lib:/opt/seres/nav_cw/deps/lib:/opt/seres/nav_cw/deps/opt/acados/lib:${THEIRS}/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LIDAR3D_MODEL_DIR="${LIDAR3D_MODEL_DIR:-/opt/seres/perception/lidar_3d_detection/model}"
export LIDAR3D_LOCK_CLOCKS="${LIDAR3D_LOCK_CLOCKS:-1}"

start_node() {
    local root=$1 name=$2 bin=$3
    local dir="${root}/${name}/lib/${name}"
    if [ ! -x "${dir}/${bin}" ]; then
        echo "[cw-nav-perception] skip ${name}: no ${dir}/${bin}" >&2
        return
    fi
    echo "[cw-nav-perception] start ${name}"
    ( cd "${dir}" && exec "./${bin}" ) &
}

start_node "${OUR}" lidar_seg lidar_seg
start_node "${OUR}" grid_map_memory_node grid_map_memory_node

ptz="${THEIRS}/ptz_data_detection/lib/ptz_data_detection"
if [ -x "${ptz}/ptz_data_detection_2d" ]; then
    echo "[cw-nav-perception] start ptz_data_detection_2d from ${THEIRS}"
    ( cd "${ptz}" && export LD_LIBRARY_PATH="${ptz}:${LD_LIBRARY_PATH}" && exec ./ptz_data_detection_2d ) &
fi
recharge="${THEIRS}/perception_recharge/lib/perception_recharge"
if [ -x "${recharge}/perception_recharge_node" ]; then
    echo "[cw-nav-perception] start perception_recharge from ${THEIRS}"
    ( cd "${recharge}" && exec ./perception_recharge_node \
        --params-file "${THEIRS}/perception_recharge/share/perception_recharge/config/perception_recharge_node.yaml" ) &
fi

if [ -x "${OUR}/lidar_3d_detection/bin/start_lidar_3d_detection.sh" ]; then
    echo "[cw-nav-perception] start lidar_3d_detection model=${LIDAR3D_MODEL_DIR} lock=${LIDAR3D_LOCK_CLOCKS}"
    "${OUR}/lidar_3d_detection/bin/lock_clocks.sh" || true
    "${OUR}/lidar_3d_detection/bin/start_lidar_3d_detection.sh" &
else
    echo "[cw-nav-perception] skip lidar_3d_detection" >&2
fi
wait
