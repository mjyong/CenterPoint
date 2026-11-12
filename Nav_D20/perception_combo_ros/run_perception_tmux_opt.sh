#!/bin/bash
# 利用 /opt/seres/nav_perception/ 下已安装的文件，通过 tmux 启动所有感知节点
# 适合 Orin 上手动调试和监控使用
# usage: bash /opt/seres/nav_perception/start/run_perception_tmux_opt.sh

PERCEPTION_INSTALL_DIR="/opt/seres/nav_perception"
ROS_SETUP="/opt/ros/humble/setup.bash"
SESSION="nav_perception"

# ============================================================
# 校验环境
# ============================================================
if ! command -v tmux &>/dev/null; then
    echo "[ERROR] tmux 未安装，请执行: sudo apt install tmux"
    exit 1
fi
if [ ! -f "${PERCEPTION_INSTALL_DIR}/setup.bash" ]; then
    echo "[ERROR] 感知环境未安装: ${PERCEPTION_INSTALL_DIR}/setup.bash 不存在"
    echo "请先执行: sudo dpkg -i nav_perception-*.deb"
    exit 1
fi

# 如果 session 已存在，直接 attach
if tmux has-session -t ${SESSION} 2>/dev/null; then
    echo "[INFO] nav_perception tmux session 已存在，直接 attach"
    tmux attach-session -t ${SESSION}
    exit 0
fi

# ============================================================
# 公共环境变量（在每个 pane 中 source）
# ============================================================
ENV_SETUP="source ${ROS_SETUP} && source ${PERCEPTION_INSTALL_DIR}/setup.bash && export LD_LIBRARY_PATH=/opt/seres/nav_perception/lib:\${LD_LIBRARY_PATH} && export SERES_PERCEPTION_INSTALL_DIR=${PERCEPTION_INSTALL_DIR}"

# ============================================================
# 创建 tmux session，窗口名: nav_perception
# ============================================================
tmux new-session -d -s ${SESSION} -n "nav_perception"

# 创建 7 个 pane（对应 6 个节点 + 1 个监控 pane）
# pane 0 已自动创建，依次切分
tmux split-window -v -t ${SESSION}:0.0     # pane 1
tmux split-window -v -t ${SESSION}:0.1     # pane 2（在 pane1 下方）
# 每行水平切分
tmux split-window -h -t ${SESSION}:0.0     # pane 0 → 0 | 3
tmux split-window -h -t ${SESSION}:0.1     # pane 1 → 1 | 4
tmux split-window -h -t ${SESSION}:0.2     # pane 2 → 2 | 5
tmux split-window -h -t ${SESSION}:0.3     # pane 3 → 3 | 6
# 使用 tiled 布局均分
tmux select-layout -t ${SESSION} tiled

# ============================================================
# Pane 0: lidar_seg
# ============================================================
tmux select-pane -t ${SESSION}:0.0
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "export LD_LIBRARY_PATH=${PERCEPTION_INSTALL_DIR}/lidar_seg/lib/lidar_seg:\${LD_LIBRARY_PATH}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/lidar_seg/lib/lidar_seg" C-m
tmux send-keys "./lidar_seg" C-m

# ============================================================
# Pane 1: grid_map_memory_node
# ============================================================
tmux select-pane -t ${SESSION}:0.1
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/grid_map_memory_node/lib/grid_map_memory_node" C-m
tmux send-keys "./grid_map_memory_node" C-m

# ============================================================
# Pane 2: pedestrian_code_conduct
# ============================================================
tmux select-pane -t ${SESSION}:0.2
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "export LD_LIBRARY_PATH=${PERCEPTION_INSTALL_DIR}/pedestrian_code_conduct/lib/pedestrian_code_conduct:\${LD_LIBRARY_PATH}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/pedestrian_code_conduct/lib/pedestrian_code_conduct" C-m
tmux send-keys "./pedestrian_code_conduct_2d" C-m

# ============================================================
# Pane 3: visual_perception
# ============================================================
tmux select-pane -t ${SESSION}:0.3
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "export LD_LIBRARY_PATH=${PERCEPTION_INSTALL_DIR}/visual_perception/lib/visual_perception:\${LD_LIBRARY_PATH}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/visual_perception/lib/visual_perception" C-m
tmux send-keys "./visual_perception_2d" C-m

# ============================================================
# Pane 4: ptz_data_detection
# ============================================================
tmux select-pane -t ${SESSION}:0.4
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "export LD_LIBRARY_PATH=${PERCEPTION_INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection:\${LD_LIBRARY_PATH}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection" C-m
tmux send-keys "./ptz_data_detection_2d" C-m

# ============================================================
# Pane 5: 监控 pane（不启动节点，留作查看日志/调试）
# ============================================================
tmux select-pane -t ${SESSION}:0.5
tmux send-keys "echo '=== Perception Monitor Pane ===' " C-m
tmux send-keys "echo 'Tips:'" C-m
tmux send-keys "echo '  ros2 topic list         -- 查看所有 topic'" C-m
tmux send-keys "echo '  ros2 node list          -- 查看所有节点'" C-m
tmux send-keys "echo '  tmux kill-session -t ${SESSION}  -- 停止所有节点'" C-m
tmux send-keys "${ENV_SETUP}" C-m

# ============================================================
# Pane 6: perception_recharge
# ============================================================
tmux select-pane -t ${SESSION}:0.6
tmux send-keys "${ENV_SETUP}" C-m
tmux send-keys "cd ${PERCEPTION_INSTALL_DIR}/perception_recharge/lib/perception_recharge" C-m
tmux send-keys "./perception_recharge_node --params-file ${PERCEPTION_INSTALL_DIR}/perception_recharge/share/perception_recharge/config/perception_recharge_node.yaml" C-m

# ============================================================
# 设置 pane 标题（便于识别）
# ============================================================
tmux select-pane -t ${SESSION}:0.0 -T "lidar_seg"
tmux select-pane -t ${SESSION}:0.1 -T "grid_map"
tmux select-pane -t ${SESSION}:0.2 -T "pedestrian"
tmux select-pane -t ${SESSION}:0.3 -T "visual_perception"
tmux select-pane -t ${SESSION}:0.4 -T "ptz_detection"
tmux select-pane -t ${SESSION}:0.5 -T "monitor"
tmux select-pane -t ${SESSION}:0.6 -T "recharge"

# 激活监控 pane
tmux select-pane -t ${SESSION}:0.5

echo "[INFO] nav_perception tmux session 已启动"
echo "[INFO] attach 命令: tmux attach-session -t ${SESSION}"
echo "[INFO] 停止所有节点: tmux kill-session -t ${SESSION}"

# 自动 attach（如果是交互终端）
if [ -t 1 ]; then
    tmux attach-session -t ${SESSION}
fi
