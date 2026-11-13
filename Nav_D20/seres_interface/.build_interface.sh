#!/bin/bash
function log_error() {
    echo -e "\E[1;31m[Error] $*\033[0m"
}

function log_ok() {
    echo -e "\E[1;32m[ OK ] $*\033[0m"
}

ros_bash="/opt/ros/humble/setup.bash"
if [ ! -f $ros_bash ];then log_error "$ros_bash not exist!!!"; exit 1; fi
source $ros_bash

pkg_list=(
    seres_proto
    hmi_interface 
    iot_interface
    mc_interface 
    robot_interface 
    sysapp_interface 
    voice_interface 
    light_interface 
    scene_interface 
    nav_interface 
    rte_interface 
    tau_interface
    media_interface
)

failed_pkg_list=(

)

for pkg_name in ${pkg_list[@]}; do
    colcon build --packages-select $pkg_name
    if [ $? -ne 0 ];then 
        log_error "[$pkg_name] - Build failed!!!"; 
        failed_pkg_list+=(${pkg_name})
    else
        log_ok "[$pkg_name] - Build success!"
    fi
done
fnumber=${#failed_pkg_list[@]}
if [ ${fnumber} -ne 0 ];then
    log_error "Build failed package list[${fnumber}]: ${failed_pkg_list[*]}"
    exit 1
fi

log_ok "Build ros package ${pkg_list[*]} all success!"
exit 0

