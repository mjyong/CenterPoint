#ifndef SERES_PERCEPTION_PTZDATADETECTION_H__
#define SERES_PERCEPTION_PTZDATADETECTION_H__

#include "rclcpp/rclcpp.hpp"
#include "../include/logging.h"
#include "std_msgs/msg/string.hpp"
#include <cmath> 
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
//
#include <nav_interface/msg/ptz_rgb_img_data.hpp>
#include <nav_interface/msg/ptz_infrard_img_data.hpp>
#include <nav_interface/msg/fine_tuning_ptz.hpp>
#include <nav_interface/msg/det_seg_results.hpp>
#include <nav_interface/msg/visual2_d.hpp>
#include <nav_interface/msg/target_ptz.hpp>

#include <perception_interface/srv/perception_event.hpp>
#include <perception_interface/msg/perception_event.hpp>
#include <perception_interface/msg/event_info.hpp>
#include <common_interface/msg/bounding_box2_d.hpp>
#include <rte_interface/msg/nav_ptz_take_photo_ctrl_multi.hpp>
#include <rte_interface/msg/nav_ptz_take_photo_data.hpp>
#include <rte_interface/msg/cam_tem_matrix.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "../yolov8_detect/yolov8.hpp"
#include "../yolov8_detect/common.hpp"

namespace seres {
namespace perception {



struct PtzDataDetectionParameter
{
   bool vis_flag;
   std::string img_topic;
   std::string infrared_topic;
   std::string to_planning_topic;
   std::string to_ptz_control_topic;
   std::string save_task_img_path;
   std::string det_seg_ret_topic;
   bool save_img;
   bool use_detection;
   std::string target_ptz_topic;
   std::string det_engine_file_path_;
   int wait_count;
   bool use_offline_mode;
   std::string offline_img_file_path;
   double center_img_x1,center_img_y1,center_img_x2,center_img_y2,center_img_x3,center_img_y3,center_img_x4,center_img_y4;
   std::string perception_detect_srv;
   std::string take_photo_ctrl_topic;
   // 货车停放（event 19）连拍：张数和相邻两张的间隔
   int multi_photo_num = 5;
   int multi_photo_interval_ms = 20;
   std::string take_photo_data_topic;
   std::string cam_tem_matrix_topic;
};




class PtzDataDetectionTestPeer;

class PtzDataDetection
{
    friend class PtzDataDetectionTestPeer;
public:
    // 构造函数
    PtzDataDetection();
    bool Init();
    void Process();
    void Run();


private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr service_callback_group_;
    rclcpp::CallbackGroup::SharedPtr sub_callback_group_;
    // std::shared_ptr<ros_ipc_com::NodeBase> ipc_node_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_,pub_infrared_;

    // // IPC publish
    // std::shared_ptr<ros_ipc_com::PublisherBase<seres_ipc::nav_interface::msg::FineTuningPtz>> m_pub_ipc_planning,m_pub_ipc_ptz_control;

    // // IPC SUB
    // std::shared_ptr<ros_ipc_com::SubscriberBase> m_rgb_subscriber;

    // std::shared_ptr<ros_ipc_com::SubscriberBase> m_infrared_subscriber;
    
    // void IPCRGBCallback(const std::shared_ptr<seres_ipc::nav_interface::msg::NavPedestrianStandard> msg);
    // void IPCInfraredCallback(const std::shared_ptr<seres_ipc::nav_interface::msg::PtzInfraredImgData> msg);


    // ROS Publish
    rclcpp::Publisher<nav_interface::msg::FineTuningPTZ>::SharedPtr m_pub_ros_planning,m_pub_ros_ptz_control;
    rclcpp::Subscription<nav_interface::msg::DetSegResults>::SharedPtr sub_det_seg_;
    rclcpp::Subscription<nav_interface::msg::TargetPTZ>::SharedPtr sud_target_ptz_;


    // ROS SUB
    rclcpp::Subscription<nav_interface::msg::PtzRGBImgData>::SharedPtr sub_ptz_img;
    rclcpp::Subscription<nav_interface::msg::PtzInfrardImgData>::SharedPtr sub_ptz_infrared;

    void ROSRGBCallback(const nav_interface::msg::PtzRGBImgData::SharedPtr msg);
    void ROSInfraredCallback(const nav_interface::msg::PtzInfrardImgData::SharedPtr msg);

    void DetSegCallback(const nav_interface::msg::DetSegResults::SharedPtr msg);
    void TargetPTZCallback(const nav_interface::msg::TargetPTZ::SharedPtr msg);
    void RetsPub();

    // Perception detect service
    rclcpp::Service<perception_interface::srv::PerceptionEvent>::SharedPtr perception_detect_srv_;
    void HandlePerceptionDetect(
        const std::shared_ptr<perception_interface::srv::PerceptionEvent::Request> request,
        std::shared_ptr<perception_interface::srv::PerceptionEvent::Response> response);

    // 云台驱动在 take_photo_ctrl_topic 上订阅 NavPtzTakePhotoCtrlMulti。
    // take_photo_id: 5 彩色，6 红外。单张时 num=1。
    rclcpp::Publisher<rte_interface::msg::NavPtzTakePhotoCtrlMulti>::SharedPtr pub_take_photo_ctrl_multi_;
    rclcpp::Subscription<rte_interface::msg::NavPtzTakePhotoData>::SharedPtr sub_take_photo_data_;
    rclcpp::Subscription<rte_interface::msg::CamTemMatrix>::SharedPtr sub_cam_tem_matrix_;

    void TakePhotoDataCallback(const rte_interface::msg::NavPtzTakePhotoData::SharedPtr msg);
    void CamTemMatrixCallback(const rte_interface::msg::CamTemMatrix::SharedPtr msg);

    bool CheckOcclusion(const cv::Mat& rgb_img);
  
private:


    bool vis_flag = true;
    std::string vis_frame_id = "dog";

    PtzDataDetectionParameter param;
    bool det_save_img;
    int wait_count = 0;
    TargetPTZData task_target_ptz;

    std::shared_ptr<YOLOv8> yolov8_;
};


} // namespace perception
} // namespace seres

#endif
