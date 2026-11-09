#!/bin/bash
# 感知节点后台批量启动脚本（供 nav_perception.service 的 ExecStart 使用）
# 特点：不依赖 tmux；所有子进程属于同一 cgroup，systemd stop 时统一终止
# 日志统一通过 journald 收集（StandardOutput=journal）

PERCEPTION_INSTALL_DIR="/opt/seres/nav_perception"
ROS_SETUP="/opt/ros/humble/setup.bash"

# ============================================================
# source ROS2 环境
# ============================================================
if [ ! -f "${ROS_SETUP}" ]; then
    echo "[nav_perception] ERROR: ROS2 setup 文件不存在: ${ROS_SETUP}" >&2
    exit 1
fi
source "${ROS_SETUP}"

if [ ! -f "${PERCEPTION_INSTALL_DIR}/setup.bash" ]; then
    echo "[nav_perception] ERROR: 感知环境 setup 文件不存在: ${PERCEPTION_INSTALL_DIR}/setup.bash" >&2
    exit 1
fi
source "${PERCEPTION_INSTALL_DIR}/setup.bash"

# ============================================================
# LD_LIBRARY_PATH: 三方依赖库（gflags / glog / octomap）
# ============================================================
# OpenCV 4.10 在 /usr/local/lib，不在系统默认搜索路径里。
# 不加上时 visual / ptz / pedestrian / grid_map 会因 libopencv_imgcodecs.so.410 立刻退出。
export LD_LIBRARY_PATH="/opt/seres/nav_perception/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 节点源码通过该环境变量定位配置文件路径（grid_map_memory_node 等必须）
export SERES_PERCEPTION_INSTALL_DIR="${PERCEPTION_INSTALL_DIR}"

echo "[nav_perception] 启动感知节点，安装目录: ${PERCEPTION_INSTALL_DIR}"

# ============================================================
# 辅助函数：启动单个节点，标准输出带节点名前缀
# ============================================================
start_node() {
    local node_name=$1
    local node_dir="${PERCEPTION_INSTALL_DIR}/${node_name}/lib/${node_name}"
    local node_bin=$2     # 可执行文件名
    local extra_env=$3    # 额外环境变量（可选）

    if [ ! -d "${node_dir}" ]; then
        echo "[nav_perception] WARN: 节点目录不存在，跳过: ${node_dir}" >&2
        return
    fi
    if [ ! -f "${node_dir}/${node_bin}" ]; then
        echo "[nav_perception] WARN: 可执行文件不存在，跳过: ${node_dir}/${node_bin}" >&2
        return
    fi

    echo "[nav_perception] 启动 ${node_name} (${node_bin})..."
    (
        cd "${node_dir}"
        [ -n "${extra_env}" ] && eval "export ${extra_env}"
        exec "./${node_bin}"
    ) &
}

# ============================================================
# 启动各感知节点
# ============================================================

# 1. lidar_seg
start_node "lidar_seg" "lidar_seg"

# 1b. lidar_3d_detection（点云 3D 检测跟踪；model 在包内 lidar_3d_detection/model）
start_node "lidar_3d_detection" "lidar_3d_detection"

# 2. grid_map_memory_node
start_node "grid_map_memory_node" "grid_map_memory_node"

# 3. visual_perception
(
    node_dir="${PERCEPTION_INSTALL_DIR}/visual_perception/lib/visual_perception"
    cd "${node_dir}"
    export LD_LIBRARY_PATH="${node_dir}:${LD_LIBRARY_PATH}"
    exec "./visual_perception_2d"
) &

# 4. ptz_data_detection
(
    node_dir="${PERCEPTION_INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection"
    cd "${node_dir}"
    export LD_LIBRARY_PATH="${node_dir}:${LD_LIBRARY_PATH}"
    exec "./ptz_data_detection_2d"
) &

# 5. pedestrian_code_conduct
(
    node_dir="${PERCEPTION_INSTALL_DIR}/pedestrian_code_conduct/lib/pedestrian_code_conduct"
    cd "${node_dir}"
    export LD_LIBRARY_PATH="${node_dir}:${LD_LIBRARY_PATH}"
    exec "./pedestrian_code_conduct_2d"
) &

echo "[nav_perception] 所有节点已在后台启动，PID 组: $$"
echo "[nav_perception] 查看日志: journalctl -u nav_perception.service -f"

# wait 使脚本保持运行，systemd 将此进程视为 service 主进程
# 当 systemd stop 时，KillMode=control-group 会终止整个进程组
wait
