
#include <string>
#include <unistd.h>
#include "../include/ptz_data_detection.h"
#include "../include/logging.h"

int main(int argc, char **argv)
{
    // 初始化 glog
    seres::perception::PtzLogger::Init();
    
    rclcpp::init(argc, argv);
    
 
    auto node = std::make_shared<seres::perception::PtzDataDetection>();
    if(!node->Init()){
        LOG(ERROR) << "ERROR Init PtzDataDetection";
        seres::perception::PtzLogger::Shutdown();
        rclcpp::shutdown();
        return -1;
    }
    node->Run();

    seres::perception::PtzLogger::Shutdown();
    rclcpp::shutdown();
    return 0;
}