#ifndef SERES_PERCEPTION_BEHAVIORDETECTION_H__
#define SERES_PERCEPTION_BEHAVIORDETECTION_H__

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include <cmath> 
#include "../include/logging.h"
#include "../include/datapool.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <thread>
#include <ctime> 
#include <nlohmann/json.hpp>
#include <filesystem>   
#include <Eigen/Dense>
#include <fstream> 
#include <atomic>
//
#include <nav_interface/msg/det_seg_results.hpp>
#include <nav_interface/msg/visual2_d.hpp>
#include <nav_interface/msg/nav_pedestrian_standard.hpp>
#include <perception_interface/action/perception_event.hpp>
#include <perception_interface/msg/perception_event.hpp>

namespace seres {
namespace perception {


const std::vector<std::string> seres_seg_2cls = {
            "zebra_crossing","walkable_area"
};

const std::vector<std::string> seres_det_3cls = {
    "Motor vehicle",
    "Pedestrian",
    "Other obstacles",
    "non-motoriezed vehicle"};

const std::vector<cv::Scalar> color3{
            cv::Scalar(207, 0, 0),cv::Scalar(250, 191, 0),cv::Scalar(150, 150, 150)
        };

struct BehaviorDetectionParameter
{
   bool vis_flag;
   std::string img_topic;
   std::string det_seg_ret_topic;
   int seg_src_h;
   int seg_src_w;
   int det_src_h;
   int det_src_w;
   int det_width_min;
   int det_width_max;
   int det_height_min;
   int det_height_max;
   double wh_ratio;
   int min_zebra_area;
   int min_walkable_area;
   bool use_offline_mode = false;
   std::string offline_mode_save_path = "none";
   std::string offline_img_dir = "none";
   double report_internal;
   std::string save_task_img_path;
   bool save_img;
   std::string perception_detect_action = "/perception/perception_detect_cycle";
};

using PerceptionDetectAction = perception_interface::action::PerceptionEvent;
using GoalHandlePerceptionDetect = rclcpp_action::ServerGoalHandle<PerceptionDetectAction>;

class BehaviorDetection
{
public:
    // 构造函数
    BehaviorDetection();
    bool Init();
    void Process();
    void Run();


private:
    rclcpp::Node::SharedPtr node_;
    // std::shared_ptr<ros_ipc_com::NodeBase> ipc_node_;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_subscribe_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr unimg_subscribe_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
    rclcpp::Subscription<nav_interface::msg::DetSegResults>::SharedPtr sub_det_seg_;

    // IPC publish
    // std::shared_ptr<ros_ipc_com::PublisherBase<seres_ipc::nav_interface::msg::NavPedestrianStandard>> m_pub_ipc_results;

    //ROS PUB
    rclcpp::Publisher<nav_interface::msg::NavPedestrianStandard>::SharedPtr m_pub_ros_results;

    
    void ImgCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void UnImgCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void DetSegCallback(const nav_interface::msg::DetSegResults::SharedPtr msg);
    void RetsPub();

    rclcpp_action::GoalResponse HandleGoal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const PerceptionDetectAction::Goal> goal);
    rclcpp_action::CancelResponse HandleCancel(
        const std::shared_ptr<GoalHandlePerceptionDetect> goal_handle);
    void HandleAccepted(
        const std::shared_ptr<GoalHandlePerceptionDetect> goal_handle);
    void ExecuteGoal(const std::shared_ptr<GoalHandlePerceptionDetect> goal_handle);
    void ResetDetectionState();
    void PublishActionFeedback(const cv::Mat & event_img);

private:
    std::mutex callback_mutex_;
    std::mutex publish_mutex_;

    float time_thresh = 2000.0;
    float IOU_thresh = 0.5;
    float IOU_face_thresh = 0.02;

    bool vis_flag = true;
    std::string vis_frame_id = "dog";
    int internal_count_ = 0;
    BehaviorDetectionParameter param;

    rclcpp_action::Server<PerceptionDetectAction>::SharedPtr action_server_;
    std::shared_ptr<GoalHandlePerceptionDetect> current_goal_handle_;
    std::mutex goal_mutex_;
    std::thread goal_thread_;
    std::atomic<bool> cycle_detect_enabled_{false};
    std::atomic<bool> cancel_requested_{false};
    uint32_t detect_cycle_ms_{0};
};


} // namespace perception
} // namespace seres

#endif
