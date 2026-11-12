#!/bin/bash
source /opt/ros/humble/setup.bash
colcon build --packages-select common_interface nav_interface rte_interface  mc_interface robot_interface
if [ $? -ne 0 ];then echo "[Error] Build result is failed!!!"; exit 1; fi
echo "[Info] $CI_JOB_NAME done!"
exit 0

