
#include <string>
#include <unistd.h>
#include "../include/visualperception.h"
#include "../include/logging.h"

int main(int argc, char **argv)
{
    // 初始化 glog
    seres::perception::Logger::Init();
    
    rclcpp::init(argc, argv);
    auto node = std::make_shared<seres::perception::VisualPerception>();
    if(!node->Init()){
        LOG(ERROR) << "ERROR Init visualperception";
        seres::perception::Logger::Shutdown();
        rclcpp::shutdown();
        return -1;
    }
    // while (rclcpp::ok()){
    //     node->Process();
    // }
    rclcpp::WallRate loop_rate(30);  // 30 Hz
    while (rclcpp::ok()){
        node->Run();
        loop_rate.sleep();
    }
    seres::perception::Logger::Shutdown();
    rclcpp::shutdown();
    return 0;
}