#!/bin/bash
# 0. export基础环境
script_abs_file_path=$(realpath "$0")
script_abs_dir_path=$(dirname ${script_abs_file_path})
export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path

# export SERES_PERCEPTION_ROOT_DIR=/home/nvidia/hhl/perception_combo
export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install
export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${LD_LIBRARY_PATH}
#export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml
echo "perception build script path : $script_abs_file_path"
echo "SERES_PERCEPTION_ROOT_DIR: $SERES_PERCEPTION_ROOT_DIR"

# lidar_seg 


# 创建3x3网格布局
tmux new-session -d -s perception

# 创建3行
tmux split-window -v -t perception:0.0
tmux split-window -v -t perception:0.1

# 每行创建3列
for row in 0 1 2; do
    tmux split-window -h -t perception:0.$row
    tmux split-window -h -t perception:0.$row
    tmux split-window -h -t perception:0.$row
done

# 设置网格布局
tmux select-layout -t perception tiled

tmux select-pane -t 0
tmux send-keys "export export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${LD_LIBRARY_PATH}" C-m
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m

tmux send-keys "cd /opt/seres/package/nav/ptz_control" C-m
tmux send-keys "source /opt/seres/package/nav/setup.bash" C-m
tmux send-keys "source /opt/seres/package/robot/setup.bash" C-m
#tmux send-keys "./IPCMaster ${SERES_PERCEPTION_ROOT_DIR}/third_party/MET_IPC_Config_1204.json" C-m


tmux select-pane -t 1
tmux send-keys "export export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${LD_LIBRARY_PATH}" C-m
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m


tmux select-pane -t 2
tmux send-keys "export export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${LD_LIBRARY_PATH}" C-m
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/lidar_seg/lib/lidar_seg" C-m 
tmux send-keys "./lidar_seg" C-m


tmux select-pane -t 3
tmux send-keys "export export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${LD_LIBRARY_PATH}" C-m
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/grid_map_memory_node/lib/grid_map_memory_node" C-m 
tmux send-keys "./grid_map_memory_node" C-m


tmux select-pane -t 4
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/visual_perception/lib/visual_perception" C-m
tmux send-keys "export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${SERES_PERCEPTION_INSTALL_DIR}/visual_perception/lib/visual_perception:${LD_LIBRARY_PATH}" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
tmux send-keys "./visual_perception_2d" C-m

tmux select-pane -t 5
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection/" C-m
tmux send-keys "export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${SERES_PERCEPTION_INSTALL_DIR}/ptz_data_detection/lib/ptz_data_detection:${LD_LIBRARY_PATH}" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
tmux send-keys "./ptz_data_detection_2d" C-m


tmux select-pane -t 6
tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
#tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
tmux send-keys "export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/install/glog-0.6.0/lib:${SERES_PERCEPTION_ROOT_DIR}/install/octomap/lib:${LD_LIBRARY_PATH}" C-m
tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/pedestrian_code_conduct/lib/pedestrian_code_conduct" C-m
tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
tmux send-keys "./pedestrian_code_conduct_2d" C-m

# tmux select-pane -t 7
# tmux send-keys "export SERES_PERCEPTION_INSTALL_DIR=${SERES_PERCEPTION_ROOT_DIR}/install" C-m
# #tmux send-keys "export CYCLONEDDS_URI=file:///home/nvidia/cyclonedds_env.xml" C-m
# tmux send-keys "export SERES_PERCEPTION_ROOT_DIR=$script_abs_dir_path" C-m
# tmux send-keys "cd ${SERES_PERCEPTION_INSTALL_DIR}/inf_det_code_conduct/lib/inf_det_code_conduct/" C-m 
# tmux send-keys "export export LD_LIBRARY_PATH=${SERES_PERCEPTION_ROOT_DIR}/third_party/middleware/install/lib:${SERES_PERCEPTION_INSTALL_DIR}/inf_det_code_conduct/lib/inf_det_code_conduct:${LD_LIBRARY_PATH}" C-m
# tmux send-keys "source ${SERES_PERCEPTION_ROOT_DIR}/install/setup.bash" C-m
# tmux send-keys "./inf_det_code_conduct_2d" C-m
