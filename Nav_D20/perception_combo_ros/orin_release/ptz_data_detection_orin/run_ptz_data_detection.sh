#!/bin/bash
# 在 Orin 上启动已编译的 ptz_data_detection。
# 用法: ./run_ptz_data_detection.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR="${ROOT}/install"

if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "未找到 /opt/ros/humble/setup.bash，目标板需要 ROS 2 Humble。"
    exit 1
fi
if [ ! -x "${INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection/ptz_data_detection_2d" ]; then
    echo "未找到可执行文件，请确认 install 目录完整。"
    exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${INSTALL_DIR}/setup.bash"

# 配置文件路径: ${SERES_PERCEPTION_INSTALL_DIR}/ptz_data_detection/config/ptz_data_detection.json
export SERES_PERCEPTION_INSTALL_DIR="${INSTALL_DIR}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

cd "${INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection"
exec ./ptz_data_detection_2d
