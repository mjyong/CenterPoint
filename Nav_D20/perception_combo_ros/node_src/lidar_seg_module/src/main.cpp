#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl/point_types.h>
#include <fstream>
#include "Eigen/Eigen" 
#include <chrono>
#include <iomanip> // 设置输出格式
#include <atomic>
#include <thread>
#include "../include/Wildpatchworkpp.h"
#include "../include/logging.h"
#include <filesystem>


int main(int argc, char** argv) {
    // 初始化 glog
    seres::perception::Logger::Init();
    
    rclcpp::init(argc, argv);
    auto node = std::make_shared<seres::perception::WildPatchWorkpp>();
    // auto config = seres::perception::LidarSegConfigType();
    std::string currentFilePath(__FILE__);
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    LOG(INFO) << "parentPath  " << std::string(parentPath);
    
    // std::string config_path = std::string(parentPath) + "/../config/lidar_seg.yaml";
    const char* root_dir = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
    std::string perception_root = root_dir;
    std::string config_path = perception_root + "/lidar_seg/config/lidar_seg.yaml";
    // std::string config_path = "/opt/seres/package/nav/lidar_seg/config/lidar_seg.yaml";

    LOG(INFO) << " config_path  " << config_path;
    YAML::Node config = YAML::LoadFile(config_path);


    // log
    // std::string log_config_path =std::string(parentPath) + "/../../../../../../output/ros2/perception/lidar_seg/config/";
    // int ret = com_loadlog(log_config_path.c_str(), "comlog.conf");
    // if (ret != 0){
    //     std::cout << "load comlog fail" << std::endl;
    //     return -1;
    // }
    // std::string node_name = "TEST ";
    // CDEBUG_LOG("---------- TESTING ");
    // CNOTICE_LOG("this is notice log:%s", node_name.c_str());
    // CFATAL_LOG("this is FATAL log:%s", node_name.c_str());
    // CWARNING_LOG("this is WARNING log:%s", node_name.c_str());

    if(!node->Init(config)){
        LOG(ERROR) << "ERROR Init patchworkpp";
        seres::perception::Logger::Shutdown();
        rclcpp::shutdown();
        return -1;
    }

    std::atomic<bool> spin_running{true};
    std::thread spin_thread([&]() {
        while (spin_running.load() && rclcpp::ok()) {
            node->SpinOnce();
        }
    });

    rclcpp::WallRate loop_rate(50);
    while (rclcpp::ok()){
        node->Process();
        loop_rate.sleep();
    }

    spin_running.store(false);
    if (spin_thread.joinable()) {
        spin_thread.join();
    }
    seres::perception::Logger::Shutdown();
    rclcpp::shutdown();
    return 0;

}
