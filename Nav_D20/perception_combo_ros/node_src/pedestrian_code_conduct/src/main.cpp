
#include <string>
#include <unistd.h>
#include "../include/behaviordetection.h"

int main(int argc, char **argv)
{
    seres::perception::Logger::Init("/tmp/pedestrian_code_conduct_logs", "pedestrian_code_conduct_2d");

    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<seres::perception::BehaviorDetection>();
        if(!node->Init()){
            LOG(ERROR) << "Init behavior_detection_2d failed";
            rclcpp::shutdown();
            seres::perception::Logger::Shutdown();
            return -1;
        }
        rclcpp::WallRate loop_rate(30);  // 30 Hz
        while (rclcpp::ok()){
            node->Run();
            loop_rate.sleep();
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception: " << e.what();
        rclcpp::shutdown();
        seres::perception::Logger::Shutdown();
        return 1;
    }

    rclcpp::shutdown();
    seres::perception::Logger::Shutdown();
    return 0;
}
