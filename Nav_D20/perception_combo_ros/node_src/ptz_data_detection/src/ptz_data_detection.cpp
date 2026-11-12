#include "../include/ptz_data_detection.h"
#include "../include/logging.h"
#include "math.h"

#include <algorithm>

namespace seres {
namespace perception {

using json = nlohmann::json;

PtzDataDetection::PtzDataDetection(){

    node_ = std::make_shared<rclcpp::Node>("ptz_data_detection");
    // ipc_node_ = std::make_shared<ros_ipc_com::NodeBase>("ptz_data_detection");
}



bool PtzDataDetection::Init()
{

    // load config
    std::string currentFilePath(__FILE__);
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    // std::string json_config_path = std::string(parentPath) + "/../config/ptz_data_detection.json";
    const char* root_dir = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
    std::string perception_root = root_dir;
    std::string json_config_path = perception_root + "/ptz_data_detection/config/ptz_data_detection.json";
    std::ifstream ifs(json_config_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + json_config_path);
        return false;
    }
    json j; 
    ifs >> j;
    std::string det_engine_file_path_;
    try {
        j.at("vis_flag").get_to(param.vis_flag);
        j.at("img_topic").get_to(param.img_topic);
        j.at("infrared_topic").get_to(param.infrared_topic);
        j.at("to_planning_topic").get_to(param.to_planning_topic);
        j.at("to_ptz_control_topic").get_to(param.to_ptz_control_topic);
        j.at("save_task_img_path").get_to(param.save_task_img_path);
        j.at("save_img").get_to(param.save_img);
        j.at("use_detection").get_to(param.use_detection);
        j.at("det_seg_ret_topic").get_to(param.det_seg_ret_topic);
        j.at("target_ptz_topic").get_to(param.target_ptz_topic);
        j.at("det_model_path").get_to(param.det_engine_file_path_);
        j.at("wait_count").get_to(param.wait_count);
        j.at("use_offline_mode").get_to(param.use_offline_mode);
        j.at("offline_img_file_path").get_to(param.offline_img_file_path);
        j.at("center_img_x1").get_to(param.center_img_x1);
        j.at("center_img_y1").get_to(param.center_img_y1);
        j.at("center_img_x2").get_to(param.center_img_x2);
        j.at("center_img_y2").get_to(param.center_img_y2);
        j.at("center_img_x3").get_to(param.center_img_x3);
        j.at("center_img_y3").get_to(param.center_img_y3);
        j.at("center_img_x4").get_to(param.center_img_x4);
        j.at("center_img_y4").get_to(param.center_img_y4);
        j.at("perception_detect_srv").get_to(param.perception_detect_srv);
        j.at("take_photo_ctrl_topic").get_to(param.take_photo_ctrl_topic);
        param.multi_photo_num = j.value("multi_photo_num", param.multi_photo_num);
        param.multi_photo_interval_ms = j.value(
            "multi_photo_interval_ms", param.multi_photo_interval_ms);
        j.at("take_photo_data_topic").get_to(param.take_photo_data_topic);
        j.at("cam_tem_matrix_topic").get_to(param.cam_tem_matrix_topic);
        
        
        
        
        
        
        


    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
    LOG(INFO) << "vis_flag: " << param.vis_flag;
    LOG(INFO) << "img_topic: " << param.img_topic;
    LOG(INFO) << "infrared_topic: " << param.infrared_topic;
    LOG(INFO) << "to_planning_topic: " << param.to_planning_topic;
    LOG(INFO) << "to_ptz_control_topic: " << param.to_ptz_control_topic;
    LOG(INFO) << " save_task_img_path " << param.save_task_img_path;
    LOG(INFO) << " save_img " << param.save_img;
    LOG(INFO) << " use_detection " << param.use_detection;
    LOG(INFO) << " param.det_seg_ret_topic " << param.det_seg_ret_topic;
    LOG(INFO) << " param.target_ptz_topic " << param.target_ptz_topic;
    LOG(INFO) << " param.use_offline_mode " << param.use_offline_mode;
    LOG(INFO) << " param.center_img_x1 " << param.center_img_x1;
    LOG(INFO) << " param.center_img_y1 " << param.center_img_y1;
    LOG(INFO) << " param.center_img_x2 " << param.center_img_x2;
    LOG(INFO) << " param.center_img_y2 " << param.center_img_y2;
    LOG(INFO) << " param.center_img_x3 " << param.center_img_x3;
    LOG(INFO) << " param.center_img_y3 " << param.center_img_y3;
    LOG(INFO) << " param.center_img_x4 " << param.center_img_x4;
    LOG(INFO) << " param.center_img_y4 " << param.center_img_y4;
    LOG(INFO) << " param.perception_detect_srv " << param.perception_detect_srv;
    LOG(INFO) << " param.take_photo_ctrl_topic " << param.take_photo_ctrl_topic;
    LOG(INFO) << " param.multi_photo_num " << param.multi_photo_num;
    LOG(INFO) << " param.multi_photo_interval_ms " << param.multi_photo_interval_ms;
    LOG(INFO) << " param.take_photo_data_topic " << param.take_photo_data_topic;
    LOG(INFO) << " param.cam_tem_matrix_topic " << param.cam_tem_matrix_topic;


    rclcpp::QoS qos_settings(10);
    // rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();


    
    // pub_img_ = node_->create_publisher<sensor_msgs::msg::Image>("/ptz_data_detection/vis_img", qos_settings);

    // pub_infrared_ = node_->create_publisher<sensor_msgs::msg::Image>("/ptz_data_detection/vis_infrared", qos_settings);
    
    // IPC Subscribe
    
    // m_rgb_subscriber = ipc_node_->createSubscriber<seres_ipc::nav_interface::msg::PtzRGBImgData>(param.img_topic, std::bind(&PtzDataDetection::IPCRGBCallback, this, std::placeholders::_1));

    // m_rgb_subscriber = ipc_node_->createSubscriber<seres_ipc::nav_interface::msg::NavPedestrianStandard>(param.img_topic, std::bind(&PtzDataDetection::IPCRGBCallback, this, std::placeholders::_1));


    service_callback_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    sub_callback_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = sub_callback_group_;

    // sub_ptz_img = node_->create_subscription<nav_interface::msg::PtzRGBImgData>(
    //     param.img_topic, qos_settings,
    //     std::bind(&PtzDataDetection::ROSRGBCallback, this, std::placeholders::_1),
    //     sub_options);


    
    // m_infrared_subscriber = ipc_node_->createSubscriber<seres_ipc::nav_interface::msg::PtzInfraredImgData>(param.infrared_topic, std::bind(&PtzDataDetection::IPCInfraredCallback, this, std::placeholders::_1));

    // sub_ptz_infrared = node_->create_subscription<nav_interface::msg::PtzInfrardImgData>(
    //     param.infrared_topic, qos_settings,
    //     std::bind(&PtzDataDetection::ROSInfraredCallback, this, std::placeholders::_1),
    //     sub_options);

    // det result sub
    // if(param.use_detection){
    //     sub_det_seg_ = node_->create_subscription<nav_interface::msg::DetSegResults>(param.det_seg_ret_topic, qos_settings, std::bind(&PtzDataDetection::DetSegCallback, this, std::placeholders::_1)); 
    // }

    
    
    // 读取云台target ptz 
    // sud_target_ptz_ = node_->create_subscription<nav_interface::msg::TargetPTZ>(
    //     param.target_ptz_topic, qos_settings,
    //     std::bind(&PtzDataDetection::TargetPTZCallback, this, std::placeholders::_1),
    //     sub_options);

    // // IPC Publish
    // m_pub_ipc_planning = ipc_node_->CreatePublisher<seres_ipc::nav_interface::msg::FineTuningPtz>(param.to_planning_topic);

    // m_pub_ipc_ptz_control = ipc_node_->CreatePublisher<seres_ipc::nav_interface::msg::FineTuningPtz>(param.to_ptz_control_topic);

    // ROS Publish
    // m_pub_ros_planning = node_->create_publisher<nav_interface::msg::FineTuningPTZ>(param.to_planning_topic,qos_settings);
    // m_pub_ros_planning = node_->create_publisher<nav_interface::msg::FineTuningPTZ>(param.to_planning_topic,10);

    // m_pub_ros_ptz_control = node_->create_publisher<nav_interface::msg::FineTuningPTZ>(param.to_ptz_control_topic,qos_settings);

    // 云台驱动订阅的是这个话题上的 NavPtzTakePhotoCtrlMulti，不是 NavPtzTakePhotoCtrl。
    pub_take_photo_ctrl_multi_ = node_->create_publisher<rte_interface::msg::NavPtzTakePhotoCtrlMulti>(
        param.take_photo_ctrl_topic, qos_settings);

    // Perception detect service server
    perception_detect_srv_ = node_->create_service<perception_interface::srv::PerceptionEvent>(
        param.perception_detect_srv,
        std::bind(&PtzDataDetection::HandlePerceptionDetect, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        service_callback_group_);

    sub_take_photo_data_ = node_->create_subscription<rte_interface::msg::NavPtzTakePhotoData>(
        param.take_photo_data_topic, qos_settings,
        std::bind(&PtzDataDetection::TakePhotoDataCallback, this, std::placeholders::_1),
        sub_options);

    sub_cam_tem_matrix_ = node_->create_subscription<rte_interface::msg::CamTemMatrix>(
        param.cam_tem_matrix_topic, qos_settings,
        std::bind(&PtzDataDetection::CamTemMatrixCallback, this, std::placeholders::_1),
        sub_options);

    LOG(INFO) << "=== 订阅者创建完成 ===";
    if(param.use_offline_mode){
        // offline_img_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/perception/offline_image", qos_settings_default);
        DataPool* DP = DataPool::Instance();
        // 读取离线文件夹图像数据
        std::vector<cv::String> filenames;
        cv::glob(param.offline_img_file_path + "/*.jpg", filenames);
        float img_count = 0.0;
        for (const auto& filename : filenames) {
            ImageData img_data;
            cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
            img_data.time = img_count;
            img_data.img = image;
            std::filesystem::path path_obj(filename);
            std::string filename_split = path_obj.filename().string();
            LOG(INFO) << "提取的文件名: " << filename_split;

            img_data.img_name = filename_split;
            DP->main_data_.img_cache.Push(img_data);
            LOG(INFO) << " --------GET IMG DATA : " << filename;
            img_count += 0.1;
        }
        LOG(INFO) << " TOTAL IMG NUM " << img_count;
    }


    yolov8_.reset(new YOLOv8(param.det_engine_file_path_));
    yolov8_->make_pipe(true);




    return true;
}

void PtzDataDetection::TargetPTZCallback(const nav_interface::msg::TargetPTZ::SharedPtr msg){

    // std::cout << " in ptz 0" << std::endl;
    DataPool* DP = DataPool::Instance();
    TargetPTZData target_ptz_;
    target_ptz_.time = 0.0;
    target_ptz_.p = msg->p;
    target_ptz_.t = msg->t;
    target_ptz_.z = msg->z;
    target_ptz_.task_type = msg->task_type;
    // std::cout << " DP->main_data_.targetptz_cache get target_ptz.p " <<  target_ptz_.p  
    //           << " t " <<target_ptz_.t
    //           << " z " << target_ptz_.z 
    //           <<" task " <<target_ptz_.task_type
    //           << std::endl;
    DP->main_data_.targetptz_cache.Push(target_ptz_);

}
void PtzDataDetection::DetSegCallback(const nav_interface::msg::DetSegResults::SharedPtr msg){

    DataPool* DP = DataPool::Instance();
    DetSegResults det_seg_results;
    det_seg_results.time = msg->time;
    det_seg_results.img_name = msg->img_name;
    for(const auto&obj:msg->dets){
        Det2D perception_object;
        perception_object.label = obj.label;
        perception_object.confidence = obj.confidence;
        // perception_object.class_name   = seres_det_3cls[obj.label];
        if(obj.coor.size() > 3 ){
            perception_object.img_box_x = obj.coor[0];    
            perception_object.img_box_y = obj.coor[1];
            perception_object.img_box_w = obj.coor[2] - obj.coor[0];        
            perception_object.img_box_h = obj.coor[3] - obj.coor[1];        
        }else{
            // std::cout << "DetSegCallback got no det coor " << std::endl;
        }
                          
        det_seg_results.det_results.push_back(perception_object);
    }

    //====seg 
    for(const auto&obj:msg->segs){
        Seg2D perception_object;
        perception_object.label = obj.label;
        perception_object.confidence = obj.confidence;
        // perception_object.class_name  = seres_seg_3cls[obj.label];
        if(obj.coor.size() >= 6 ){
            for(int i=0;i<obj.coor.size();i+=2){
                perception_object.mask_x.push_back(obj.coor[i]);
                perception_object.mask_y.push_back(obj.coor[i+1]); 
            }       
        }else{
            // std::cout << "DetSegCallback got no seg coor " << std::endl;
        }
                          
        det_seg_results.seg_results.push_back(perception_object);
    }

    DP->main_data_.detseg_cache.Push(det_seg_results);
    // std::cout << "-- PTZ DATA Detection got ret " << det_seg_results.time 
    //           << " | img_name " << det_seg_results.img_name
    //           << " | det size " <<det_seg_results.det_results.size()
    //           << " | seg size " <<det_seg_results.seg_results.size()
    //           <<std::endl;

}

// void PtzDataDetection::IPCRGBCallback(const std::shared_ptr<seres_ipc::nav_interface::msg::NavPedestrianStandard> msg)
// {

//     try
//     {
        
//         // // 方法1: 使用clone()确保数据独立性（推荐）
//         // cv::Mat image(static_cast<int>(msg->height),
//         //              static_cast<int>(msg->width),
//         //              CV_8UC3,
//         //              const_cast<uint8_t*>(msg->data.data()),
//         //              static_cast<size_t>(msg->step));
        
//         // cv::Mat image_clone = image.clone();  // 关键：创建独立拷贝

//         // 方法2: 使用memcpy手动拷贝（替代方案）
//         std::cout << " static_cast<int>(msg->height)  " << static_cast<int>(msg->height) 
//                   << " static_cast<int>(msg->width) " << static_cast<int>(msg->width)
//                   << " msg->header.timestamp_ms " << msg->header.timestamp_ms << std::endl;
//         cv::Mat image_clone(static_cast<int>(msg->height), 
//                            static_cast<int>(msg->width), 
//                            CV_8UC3);
        
//         if (msg->data.size() == image_clone.total() * image_clone.elemSize()) {
//             std::memcpy(image_clone.data, msg->data.data(), msg->data.size());
//         } else {
//             std::cerr << "Data size mismatch" << std::endl;
//             return;
//         }

//         // 访问 cv::Mat 图像
//         // cv::Mat img = cv_ptr->image;
//         // 在此处处理图像
//         // std::unique_lock<std::mutex> locker(callback_mutex_);
//         DataPool* DP = DataPool::Instance();
//         ImageData img_data;
//         img_data.time = msg->header.timestamp_ms*1e-3;
//         img_data.img = image_clone;
//         DP->main_data_.img_cache.Push(img_data);
//         std::cout << " --------GET IMG DATA : " << img_data.time  << std::endl;

//         cv_bridge::CvImage cv_image;
//         cv_image.image = image_clone;
//         cv_image.encoding = sensor_msgs::image_encodings::BGR8;
//         auto ros_image_msg = cv_image.toImageMsg();
//         auto t3 = std::chrono::system_clock::now();

//         pub_img_->publish(*ros_image_msg);
//     }
//     catch (cv_bridge::Exception &e)
//     {
//        return;
//     }
// }

// void PtzDataDetection::ROSRGBCallback(const nav_interface::msg::PtzRGBImgData::SharedPtr msg)
// {

//     try
//     {
        
//         // // 方法1: 使用clone()确保数据独立性（推荐）
//         // cv::Mat image(static_cast<int>(msg->height),
//         //              static_cast<int>(msg->width),
//         //              CV_8UC3,
//         //              const_cast<uint8_t*>(msg->data.data()),
//         //              static_cast<size_t>(msg->step));
        
//         // cv::Mat image_clone = image.clone();  // 关键：创建独立拷贝

//         // 方法2: 使用memcpy手动拷贝（替代方案）
//         std::cout << " static_cast<int>(msg->height)  " << static_cast<int>(msg->height) 
//                   << " static_cast<int>(msg->width) " << static_cast<int>(msg->width)
//                   << " msg->header.timestamp_ms " << msg->header.timestamp_ms << std::endl;
//         cv::Mat image_clone(static_cast<int>(msg->height), 
//                            static_cast<int>(msg->width), 
//                            CV_8UC3);
        
//         if (msg->data.size() == image_clone.total() * image_clone.elemSize()) {
//             std::memcpy(image_clone.data, msg->data.data(), msg->data.size());
//         } else {
//             std::cerr << "Data size mismatch" << std::endl;
//             // return;
//         }

//         // 访问 cv::Mat 图像
//         // cv::Mat img = cv_ptr->image;
//         // 在此处处理图像
//         // std::unique_lock<std::mutex> locker(callback_mutex_);
//         DataPool* DP = DataPool::Instance();
//         ImageData img_data;
//         img_data.time = msg->header.timestamp_ms*1e-3;
//         img_data.img = image_clone;
//         DP->main_data_.img_cache.Push(img_data);
//         std::cout << std::setprecision(13)<< " --------GET IMG DATA : " << img_data.time  << std::endl;

//         cv_bridge::CvImage cv_image;
//         cv_image.image = image_clone;
//         cv_image.encoding = sensor_msgs::image_encodings::BGR8;
//         auto ros_image_msg = cv_image.toImageMsg();
//         auto t3 = std::chrono::system_clock::now();

//         pub_img_->publish(*ros_image_msg);
//     }
//     catch (cv_bridge::Exception &e)
//     {
//        return;
//     }
// }

// void PtzDataDetection::IPCInfraredCallback(const std::shared_ptr<seres_ipc::nav_interface::msg::PtzInfraredImgData> msg)
// {

//     try
//     {
        
//         // // 方法1: 使用clone()确保数据独立性（推荐）
//         // cv::Mat image(static_cast<int>(msg->height),
//         //              static_cast<int>(msg->width),
//         //              CV_8UC3,
//         //              const_cast<uint8_t*>(msg->data.data()),
//         //              static_cast<size_t>(msg->step));
        
//         // cv::Mat image_clone = image.clone();  // 关键：创建独立拷贝

//         // 方法2: 使用memcpy手动拷贝（替代方案）
//         std::cout << " static_cast<int>(msg->height)  " << static_cast<int>(msg->height) 
//                   << " static_cast<int>(msg->width) " << static_cast<int>(msg->width)
//                   << " msg->header.timestamp_ms " << msg->header.timestamp_ms 
//                   << " msg->data.size()  " << msg->data.size()
//                   << std::endl;
//         cv::Mat image_clone(static_cast<int>(msg->height), 
//                            static_cast<int>(msg->width), 
//                            CV_8UC3);
//         bool fune_tuned = msg->fine_tuing; // 微调前一定是false
//         std::cout << " Infrared " << image_clone.total() * image_clone.elemSize() << std::endl;
//         if (msg->data.size() == image_clone.total() * image_clone.elemSize() && !fune_tuned) {
//             std::memcpy(image_clone.data, msg->data.data(), msg->data.size());
//         } else {
//             std::cerr << "Data size mismatch or already fune_tuned" << std::endl;
//             return;
//         }

//         // 访问 cv::Mat 图像
//         // cv::Mat img = cv_ptr->image;
//         // 在此处处理图像
//         // std::unique_lock<std::mutex> locker(callback_mutex_);
//         DataPool* DP = DataPool::Instance();
//         ImageData img_data;
//         img_data.time = msg->header.timestamp_ms*1e-3;
//         img_data.img = image_clone;
//         DP->main_data_.infrared_cache.Push(img_data);
//         std::cout << " --------GET INFRARED DATA : " << img_data.time  << std::endl;

//         cv_bridge::CvImage cv_image;
//         cv_image.image = image_clone;
//         cv_image.encoding = sensor_msgs::image_encodings::BGR8;
//         auto ros_image_msg = cv_image.toImageMsg();
//         auto t3 = std::chrono::system_clock::now();

//         pub_infrared_->publish(*ros_image_msg);
//     }
//     catch (cv_bridge::Exception &e)
//     {
//        return;
//     }
// }

// void PtzDataDetection::ROSInfraredCallback(const nav_interface::msg::PtzInfrardImgData::SharedPtr msg)
// {

//     try
//     {
        
//         // // 方法1: 使用clone()确保数据独立性（推荐）
//         // cv::Mat image(static_cast<int>(msg->height),
//         //              static_cast<int>(msg->width),
//         //              CV_8UC3,
//         //              const_cast<uint8_t*>(msg->data.data()),
//         //              static_cast<size_t>(msg->step));
        
//         // cv::Mat image_clone = image.clone();  // 关键：创建独立拷贝

//         // 方法2: 使用memcpy手动拷贝（替代方案）
//         std::cout << " static_cast<int>(msg->height)  " << static_cast<int>(msg->height) 
//                   << " static_cast<int>(msg->width) " << static_cast<int>(msg->width)
//                   << " msg->header.timestamp_ms " << msg->header.timestamp_ms 
//                   << " msg->data.size()  " << msg->data.size()
//                   << std::endl;
//         cv::Mat image_clone(static_cast<int>(msg->height), 
//                            static_cast<int>(msg->width), 
//                            CV_8UC3);
//         // cv::Mat image_clone;


//         bool fune_tuned = msg->fine_tuing; // 微调前一定是false
//         std::cout << " Infrared " << image_clone.total() * image_clone.elemSize() << std::endl;
//         if (msg->data.size() == image_clone.total() * image_clone.elemSize() && !fune_tuned) {
//         // 只有在未微调且数据大小匹配时才创建有数据的图像
//             // image_clone = cv::Mat(static_cast<int>(msg->height), 
//             //              static_cast<int>(msg->width), 
//             //              CV_8UC3);
//             std::memcpy(image_clone.data, msg->data.data(), msg->data.size());

//             std::cerr << "GET REAL INFRARED DATA" << std::endl;
//         } else {
//             std::cerr << "Data size mismatch or already fune_tuned" << std::endl;
//             // // return;
//             // image_clone = cv::Mat();
//             // // 验证一下
//             // std::cout << "Image is empty: " << image_clone.empty() << std::endl;
//             return;
//         }

//         // 访问 cv::Mat 图像
//         // cv::Mat img = cv_ptr->image;
//         // 在此处处理图像
//         // std::unique_lock<std::mutex> locker(callback_mutex_);
//         DataPool* DP = DataPool::Instance();
//         ImageData img_data;
//         img_data.time = msg->header.timestamp_ms*1e-3;
//         img_data.img = image_clone;
//         DP->main_data_.infrared_cache.Push(img_data);
//         std::cout << std::setprecision(13) << " --------GET INFRARED DATA : " << img_data.time  << std::endl;

//         cv_bridge::CvImage cv_image;
//         cv_image.image = image_clone;
//         cv_image.encoding = sensor_msgs::image_encodings::BGR8;
//         auto ros_image_msg = cv_image.toImageMsg();
//         auto t3 = std::chrono::system_clock::now();

//         pub_infrared_->publish(*ros_image_msg);
//     }
//     catch (cv_bridge::Exception &e)
//     {
//        return;
//     }
// }

namespace {
constexpr uint32_t kTakePhotoColorChannel = 5;
constexpr uint32_t kTakePhotoInfraredChannel = 6;

cv::Mat DecodeCompressedImage(const sensor_msgs::msg::CompressedImage& msg)
{
    cv::Mat raw_data(1, static_cast<int>(msg.data.size()), CV_8UC1,
                     const_cast<uint8_t*>(msg.data.data()));
    return cv::imdecode(raw_data, cv::IMREAD_COLOR);
}
}  // namespace

void PtzDataDetection::TakePhotoDataCallback(const rte_interface::msg::NavPtzTakePhotoData::SharedPtr msg)
{
    try {
        if (msg->take_photo_id != kTakePhotoColorChannel &&
            msg->take_photo_id != kTakePhotoInfraredChannel) {
            LOG(WARNING) << "TakePhotoDataCallback: ignore take_photo_id " << msg->take_photo_id;
            return;
        }
        if (msg->images.empty()) {
            LOG(ERROR) << "TakePhotoDataCallback: empty images, take_photo_id " << msg->take_photo_id;
            return;
        }

        DataPool* DP = DataPool::Instance();
        for (const auto& image_msg : msg->images) {
            cv::Mat image = DecodeCompressedImage(image_msg);
            if (image.empty()) {
                LOG(ERROR) << "TakePhotoDataCallback: failed to decode jpeg, take_photo_id "
                           << msg->take_photo_id;
                continue;
            }
            ImageData img_data;
            const auto& stamp = (image_msg.header.stamp.sec != 0 || image_msg.header.stamp.nanosec != 0)
                                    ? image_msg.header.stamp
                                    : msg->header.stamp;
            img_data.time = stamp.sec + stamp.nanosec * 1e-9;
            img_data.img = image;
            if (msg->take_photo_id == kTakePhotoColorChannel) {
                DP->main_data_.grab_rgb_cache.Push(img_data);
            } else {
                DP->main_data_.grab_infrared_cache.Push(img_data);
            }
            LOG(INFO) << std::setprecision(13) << "--------GET TAKE PHOTO DATA id "
                      << msg->take_photo_id << " time " << img_data.time
                      << " num " << msg->num;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "TakePhotoDataCallback error: " << e.what();
    }
}

void PtzDataDetection::CamTemMatrixCallback(const rte_interface::msg::CamTemMatrix::SharedPtr msg)
{
    try {
        if (msg->is_freezedata) {
            LOG(WARNING) << "Temperature data is frozen, skipping...";
            return;
        }

        cv::Mat temp_matrix = cv::Mat(msg->height, msg->width, CV_32F,
                                      (void*)msg->data.data()).clone();

        DataPool* DP = DataPool::Instance();
        ImageData temp_data;
        temp_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        temp_data.img = temp_matrix;
        DP->main_data_.matrix_cache.Push(temp_data);
        LOG(INFO) << "--------GET TEMPERATURE DATA : " << temp_data.time
                  << " Size: " << msg->width << "x" << msg->height;
    } catch (const std::exception& e) {
        LOG(ERROR) << "CamTemMatrixCallback error: " << e.what();
    }
}

bool checkRectIntersection(const cv::Rect& rect1, const cv::Rect& rect2) {
    // 检查两个矩形是否相交
    if (rect1.x > rect2.x + rect2.width)  return false; // rect1 在 rect2 右侧
    if (rect1.x + rect1.width < rect2.x)  return false; // rect1 在 rect2 左侧
    if (rect1.y > rect2.y + rect2.height) return false; // rect1 在 rect2 下方
    if (rect1.y + rect1.height < rect2.y) return false; // rect1 在 rect2 上方
    
    return true; // 以上条件都不满足，说明相交
}
void PtzDataDetection::Run(){
    // 仅处理 service / 订阅回调时，不再后台循环调用 Process()（云台微调+YOLO 连续逻辑已停用）
    // std::thread process_thread = std::thread([this](){
    //     while (rclcpp::ok()){
    //         Process();
    //         std::this_thread::sleep_for(std::chrono::milliseconds(5));
    //     }
    // });

    // HandlePerceptionDetect 内会阻塞等待 grab 数据，需多线程 executor，否则订阅回调无法与 service 并发
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node_);
    executor.spin();

    // if (process_thread.joinable()) {
    //     process_thread.join();
    // }
}

void PtzDataDetection::Process()
{   
    auto start_process = std::chrono::system_clock::now();
    DataPool* DP = DataPool::Instance();
    
    // 每个任务点都拿RGB图像检测是否存在遮挡
    if(DP->main_data_.img_cache.GetDataSize() > 0){

        nav_interface::msg::FineTuningPTZ ros_msg;
        // 任务点时拿到target_ptz，控制云台保持朝向目标
        if(DP->main_data_.targetptz_cache.GetDataSize() > 0 || fabs(task_target_ptz.p) > 1e-100 || fabs(task_target_ptz.t) > 1e-100 || fabs(task_target_ptz.z) > 1e-100){
            DP->main_data_.targetptz_cache.PopNewest(task_target_ptz);

            // if(task_target_ptz.task_type == 2 || task_target_ptz.task_type == 4 ){
                std::cout << " DP->main_data_.targetptz_cache get target_ptz.p or have history value " <<  task_target_ptz.p  
                << " t " <<task_target_ptz.t
                << " z " << task_target_ptz.z 
                <<" task " <<task_target_ptz.task_type
                << std::endl;

                ros_msg.p = task_target_ptz.p;
                ros_msg.t = task_target_ptz.t;
                ros_msg.z = task_target_ptz.z;
                ros_msg.fine_tuing = false;

            // }else{
            //     ros_msg.p = task_target_ptz.p;
            //     ros_msg.t = task_target_ptz.t;
            //     ros_msg.z = task_target_ptz.z;
            //     ros_msg.fine_tuing = true;
            // }
            

            ImageData image_data;
            DP->main_data_.img_cache.PopNewest(image_data);
            // seres_ipc::nav_interface::msg::FineTuningPtz ipc_msg;
            
            // else if(fabs(target_ptz.p) < 1e-10 && fabs(target_ptz.t) < 1e-10 && fabs(target_ptz.z) < 1e-10){
            //     ros_msg.p = 0.0;
            //     ros_msg.t = 0.0;
            //     ros_msg.z = 0.0;
            //     ros_msg.fine_tuing = false;
            // }

            if(param.use_detection){
                std::cout << " ptz data detection use detection --------- !" << std::endl;

        

                // 拿图像进行目标检测
                cv::Size size_ = cv::Size{640, 640};
                cv::Mat img_res_;
                cv::Mat overlay_img = image_data.img;
                std::vector<Object> objs_;
                yolov8_->copy_from_Mat(image_data.img.clone(), size_);
                auto start = std::chrono::system_clock::now();
                yolov8_->infer();
                
                // 默认版本
                yolov8_->postprocess(objs_);
                yolov8_->draw_objects(overlay_img, img_res_, objs_, CLASS_NAMES, COLORS);
                auto end = std::chrono::system_clock::now();

                auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;    
                std::cout << "det Inference time: " << tc << " ms" << std::endl;

       

                // DET RESULT
                
                for(const auto& obj : objs_){
                    std::cout << "DET label: " << obj.label 
                            << " prob: " << obj.prob 
                            << " box x " << obj.rect.x 
                            << " box y " << obj.rect.y 
                            << " box width " << obj.rect.width
                            << " box height " << obj.rect.height
                            << std::endl;
                    // double center_img_x = image_data.img.cols / 2;
                    // double center_img_y = image_data.img.rows / 2;

                    double center_img_x1 = param.center_img_x1;
                    double center_img_y1 = param.center_img_y1;

                    double center_img_x2 = param.center_img_x2;
                    double center_img_y2 = param.center_img_x2;

                    double center_img_x3 = param.center_img_x3;
                    double center_img_y3 = param.center_img_y3;

                    double center_img_x4 = param.center_img_x4;
                    double center_img_y4 = param.center_img_y4;


                    cv::Rect boundingBox(obj.rect.x,obj.rect.y,obj.rect.width,obj.rect.height);
                    cv::Rect boundingBox2(center_img_x1,center_img_y1,center_img_x2-center_img_x1,center_img_y4 - center_img_y1);

                    bool in_polygon1 = boundingBox.contains(cv::Point(center_img_x1, center_img_y1));
                    bool in_polygon2 = boundingBox.contains(cv::Point(center_img_x2, center_img_y2));
                    bool in_polygon3 = boundingBox.contains(cv::Point(center_img_x3, center_img_y3));
                    bool in_polygon4 = boundingBox.contains(cv::Point(center_img_x4, center_img_y4));


                    bool intersect = checkRectIntersection(boundingBox,boundingBox2);

                    // if(in_polygon1 || in_polygon2 || in_polygon3 || in_polygon4){
                    if(intersect){
                        ros_msg.fine_tuing = false;
                        std::cout << " there is a obj in front of task target , wait ! "  << std::endl;
                        // save img 
                        if(param.save_img){
                            cv::Mat show_img = img_res_;
                            // cv::circle(show_img, cv::Point(center_img_x1, center_img_y1), 10, cv::Scalar(0, 255, 0), -1); // 绿色点
                            // cv::circle(show_img, cv::Point(center_img_x2, center_img_y2), 10, cv::Scalar(0, 255, 0), -1); // 绿色点
                            // cv::circle(show_img, cv::Point(center_img_x3, center_img_y3), 10, cv::Scalar(0, 255, 0), -1); // 绿色点
                            // cv::circle(show_img, cv::Point(center_img_x4, center_img_y4), 10, cv::Scalar(0, 255, 0), -1); // 绿色点

                            cv::rectangle(show_img, boundingBox2, cv::Scalar(0, 255, 0), 6);
                            std::string save_img_path = param.save_task_img_path + "/" + std::to_string(image_data.time) + "ptz_det.png";
                            cv::imwrite(save_img_path,show_img);
                            std::cout << " save img to " << save_img_path << std::endl;
                        }
                        break;
                    }else {
                        ros_msg.fine_tuing = true;
                        // sleep(2);
                        // std::cout << "--------------  fine_tune ! sleep "  << std::endl;
                        
                    }
                }
                if(objs_.size() < 1 ){
                    ros_msg.fine_tuing = true;
                    // sleep(2);
                    // std::cout << "--------------  fine_tune ! sleep "  << std::endl;
                }

            }else if(!image_data.img.empty()){
                ros_msg.fine_tuing = true;
                

                // save img 
                if(param.save_img ){
                    std::string save_img_path = param.save_task_img_path + "/" + std::to_string(image_data.time) + ".png";
                    cv::imwrite(save_img_path,image_data.img);
                    std::cout << " save img to " << save_img_path << std::endl;
                }

                // sleep(2);
                // std::cout << "--------------  fine_tune ! sleep "  << std::endl;
                
                

            }
            
            // 发布"seres_ipc数据"
            // m_pub_ipc_ptz_control->send_message(ipc_msg);
            if(task_target_ptz.task_type == 1){
                std::cout << "LIGTH TASK ALWAYS TRUE" << std::endl;
                ros_msg.fine_tuing = true;
            }
            // pub ros
            ros_msg.p = task_target_ptz.p;
            ros_msg.t = task_target_ptz.t;
            ros_msg.z = task_target_ptz.z;
            if(!ros_msg.fine_tuing){
                wait_count += 1;
                if(wait_count > param.wait_count ){
                    std::cout << " @@@@@@@@@@@2  wait too long, continue ! " <<std::endl;
                    ros_msg.fine_tuing = true;
                    
                    wait_count = 0;
                    // sleep(2);
                    // std::cout << "--------------  fine_tune ! sleep "  << std::endl;
                }
            }
            std::cout << " ----------- FineTuningPtz RGB  " 
                    << " p " << ros_msg.p
                    << " t " <<ros_msg.t
                    << " z " <<ros_msg.z
                    << " fine tune " << ros_msg.fine_tuing << std::endl;
            


            if(ros_msg.fine_tuing){
                std::cout << " === > RESET task_target_ptz " << std::endl;
                task_target_ptz.p = 0.0;
                task_target_ptz.t = 0.0;
                task_target_ptz.z = 0.0;

                if(param.save_img){
                            
                    std::string save_img_path = param.save_task_img_path + "/" + std::to_string(image_data.time) + "ptz_good.png";
                    cv::imwrite(save_img_path,image_data.img);
                    std::cout << " save img to " << save_img_path << std::endl;
                }
                // sleep(1);
                // std::cout << "--------------  fine_tune ! sleep "  << std::endl;
            }
            // if(ros_msg){
            //     sleep(2);
            //     std::cout << "--------------  fine_tune ! sleep "  << std::endl;
            // }

            m_pub_ros_ptz_control->publish(ros_msg);
        }
        // else if(fabs(task_target_ptz.p) < 1e-100 && fabs(task_target_ptz.t) < 1e-100 && fabs(task_target_ptz.z) < 1e-100){
        //     ros_msg.p = 0.0;
        //     ros_msg.t = 0.0;
        //     ros_msg.z = 0.0;
        //     ros_msg.fine_tuing = false;
        // }

        
    }

    // if(DP->main_data_.infrared_cache.GetDataSize() > 0){
    //     ImageData image_data;
    //     DP->main_data_.infrared_cache.PopNewest(image_data);
    //     // nav_interface::msg::FineTuningPTZ ros_msg;
    //     if(DP->main_data_.targetptz_cache.GetDataSize() > 0){

    //         DP->main_data_.targetptz_cache.PopNewest(target_ptz);

    //         std::cout << " DP->main_data_.targetptz_cache get target_ptz.p " <<  target_ptz.p  
    //           << " t " <<target_ptz.t
    //           << " z " << target_ptz.z 
    //           <<" task " <<target_ptz.task_type
    //           << std::endl;

    //         ros_msg.p = target_ptz.p;
    //         ros_msg.t = target_ptz.t;
    //         ros_msg.z = target_ptz.z;
    //         ros_msg.fine_tuing = false;
    //     }
    //     // else if(fabs(target_ptz.p) < 1e-10 && fabs(target_ptz.t) < 1e-10 && fabs(target_ptz.z) < 1e-10){
    //     //     ros_msg.p = 0.0;
    //     //     ros_msg.t = 0.0;
    //     //     ros_msg.z = 0.0;
    //     //     ros_msg.fine_tuing = false;
    //     // }

    //     if(param.use_detection){
    //         std::cout << " ptz data infrared detection use detection --------- !" << std::endl;

    //         det_save_img = false;
    //         if(DP->main_data_.detseg_cache.GetDataSize() > 0){
    //             DetSegResults det_seg_results;
    //             DP->main_data_.detseg_cache.PopNewest(det_seg_results); //先拿最旧的数据处理
    //             std::cout << std::setprecision(13) <<  "====== det_seg_results in ok time "  << det_seg_results.time << std::endl;
    //             if(det_seg_results.det_results.size() > 0){
    //                 std::cout << " GOT DET IN IMG " << std::endl;
    //                 for(const auto&obj:det_seg_results.det_results){
    //                     // if(obj.label == 1 || obj.label == 4 || obj.label == ){  // person or car                         
    //                     // }
    //                     std::cout << "DET obj.img_box_x : " << obj.img_box_x
    //                           << "DET obj.img_box_y : " << obj.img_box_y
    //                           << "DET obj.img_box_w : " << obj.img_box_w 
    //                           << "DET obj.img_box_H : " << obj.img_box_h 
    //                           << " DET label : " <<  obj.label 
    //                           << std::endl;
    //                     double center_img_x = 1920 / 2;
    //                     double center_img_y = 1080 / 2;
    //                     cv::Rect boundingBox(obj.img_box_x,obj.img_box_y,obj.img_box_w,obj.img_box_h);
    //                     bool in_polygon = boundingBox.contains(cv::Point(center_img_x, center_img_y));
                        
                        
    //                     if(in_polygon){
    //                         ros_msg.fine_tuing = false;
    //                         std::cout << " there is a obj in front of infrared task target , wait ! "  << std::endl;
    //                         break;
    //                     }else{
    //                         ros_msg.fine_tuing = true;
    //                     }

                        
    //                 }
    //             }
    //         }else if(!image_data.img.empty()){
    //             ros_msg.fine_tuing = true;
    //             if(param.save_img ){
    //                 std::string save_img_path = param.save_task_img_path + "/" + std::to_string(image_data.time) + "infrared.png";
    //                 cv::imwrite(save_img_path,image_data.img);
    //                 std::cout << " save img to " << save_img_path << std::endl;
    //             }
    //         }
    //     }else if(!image_data.img.empty()){
    //         ros_msg.fine_tuing = true;

    //         // save img 
    //         if(param.save_img ){
    //             std::string save_img_path = param.save_task_img_path + "/" + std::to_string(image_data.time) + "infrared.png";
    //             cv::imwrite(save_img_path,image_data.img);
    //             std::cout << " save img to " << save_img_path << std::endl;
    //         }
            
            

    //     }
        
    //     // 发布"seres_ipc数据"
    //     // m_pub_ipc_ptz_control->send_message(ipc_msg);

    //     // pub ros
    //     ros_msg.p = target_ptz.p;
    //     ros_msg.t = target_ptz.t;
    //     ros_msg.z = target_ptz.z;
    //     if(!ros_msg.fine_tuing){
    //         wait_count += 1;
    //         if(wait_count > 20 ){
    //             std::cout << " @@@@@@@@@@@2  wait too long, continue ! " <<std::endl;
    //             ros_msg.fine_tuing = true;
    //             wait_count = 0;
    //         }
    //     }

    //     std::cout << " ----------- FineTuningPtz INFRARED  " 
    //               << " p " << ros_msg.p
    //               << " t " <<ros_msg.t
    //               << " z " <<ros_msg.z
    //               << " fine tune " << ros_msg.fine_tuing << std::endl;

    //     m_pub_ros_ptz_control->publish(ros_msg);

    // }

    return;
    
}

void PtzDataDetection::RetsPub(){

    return;
}

bool PtzDataDetection::CheckOcclusion(const cv::Mat& rgb_img)
{
    if (!param.use_detection || rgb_img.empty()) {
        return false;
    }

    cv::Size size_ = cv::Size{640, 640};
    cv::Mat img_res_;
    cv::Mat overlay_img = rgb_img.clone();
    std::vector<Object> objs_;
    yolov8_->copy_from_Mat(rgb_img.clone(), size_);
    yolov8_->infer();
    yolov8_->postprocess(objs_);
    yolov8_->draw_objects(overlay_img, img_res_, objs_, CLASS_NAMES, COLORS);

    double center_img_x1 = param.center_img_x1;
    double center_img_y1 = param.center_img_y1;
    double center_img_x2 = param.center_img_x2;
    double center_img_y4 = param.center_img_y4;

    cv::Rect boundingBox2(center_img_x1, center_img_y1,
                          center_img_x2 - center_img_x1,
                          center_img_y4 - center_img_y1);

    for (const auto& obj : objs_) {
        cv::Rect boundingBox(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        if (checkRectIntersection(boundingBox, boundingBox2)) {
            LOG(WARNING) << "CheckOcclusion: object detected in front of target, occluded";
            if (param.save_img) {
                cv::rectangle(img_res_, boundingBox2, cv::Scalar(0, 255, 0), 6);
                auto now = std::chrono::system_clock::now();
                auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                std::string save_path = param.save_task_img_path + "/" + std::to_string(ts) + "_occluded.png";
                cv::imwrite(save_path, img_res_);
            }
            return true;
        }
    }
    return false;
}

void PtzDataDetection::HandlePerceptionDetect(
    const std::shared_ptr<perception_interface::srv::PerceptionEvent::Request> request,
    std::shared_ptr<perception_interface::srv::PerceptionEvent::Response> response)
{
    uint32_t event_id = request->percept.event_id;
    LOG(INFO) << "HandlePerceptionDetect received event_id: " << event_id;

    const uint32_t INFRARED_HIGH_TEMPERATURE_DETECTION = 13;
    const uint32_t BUILDING_APPEARANCE_CHANGE_DETECTION = 14;
    const uint32_t STREET_LIGHT_WORKING_STATUS_DETECTION = 15;
    const uint32_t TRUCK_PARKING_DETECTION =
        perception_interface::msg::PerceptionEvent::TRUCK_PARKING_DETECTION;

    bool is_infrared_task = (event_id == INFRARED_HIGH_TEMPERATURE_DETECTION);
    bool is_rgb_task = (event_id == BUILDING_APPEARANCE_CHANGE_DETECTION ||
                        event_id == STREET_LIGHT_WORKING_STATUS_DETECTION);
    bool is_truck_parking = (event_id == TRUCK_PARKING_DETECTION);

    auto make_event_info = [this]() {
        perception_interface::msg::EventInfo info;
        info.stamp = node_->get_clock()->now();
        return info;
    };
    auto push_error = [&](const std::string& err_msg) {
        auto info = make_event_info();
        info.object.format = "error";
        info.object.data.assign(err_msg.begin(), err_msg.end());
        response->event_info.push_back(info);
    };

    if (!is_infrared_task && !is_rgb_task && !is_truck_parking) {
        LOG(ERROR) << "HandlePerceptionDetect: unsupported event_id " << event_id;
        push_error("unsupported event_id: " + std::to_string(event_id)
            + " (supported: 13=infrared, 14=building, 15=street_light, 19=truck_parking)");
        return;
    }

    DataPool* DP = DataPool::Instance();
    const int max_poll = param.wait_count > 0 ? param.wait_count : 50;
    const int poll_interval_ms = 100;
    const int interval_ms = param.multi_photo_interval_ms >= 0 ? param.multi_photo_interval_ms : 20;

    if (!pub_take_photo_ctrl_multi_) {
        LOG(ERROR) << "HandlePerceptionDetect: multi photo publisher is not ready";
        push_error("multi photo publisher is not ready");
        return;
    }

    auto publish_take_photo = [this](uint32_t channel_id, uint32_t num, uint32_t interval) {
        rte_interface::msg::NavPtzTakePhotoCtrlMulti ctrl_msg;
        ctrl_msg.header.stamp = node_->get_clock()->now();
        ctrl_msg.take_photo_id = channel_id;
        ctrl_msg.num = num;
        ctrl_msg.time_interval = interval;
        pub_take_photo_ctrl_multi_->publish(ctrl_msg);
        LOG(INFO) << "Published NavPtzTakePhotoCtrlMulti topic take_photo_id=" << channel_id
                  << " num=" << num << " time_interval_ms=" << interval;
    };

    // 云台同时只处理一条拍照请求，上一条还没结束时后一条会被丢掉。
    // 一次只发一个通道，回图后再发下一个；超时则隔开重发。
    constexpr int kMaxSendAttempts = 3;
    constexpr int kResendGapMs = 300;
    constexpr int kChannelGapMs = 200;

    auto sleep_ms = [](int ms) {
        if (ms > 0 && rclcpp::ok()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    };

    auto wait_until = [&](const auto& ready, int polls, int attempt, const char* name) {
        if (ready()) {
            return true;
        }
        for (int poll_count = 1; poll_count <= polls && rclcpp::ok(); ++poll_count) {
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
            const int rgb_cache_size = DP->main_data_.grab_rgb_cache.GetDataSize();
            const int infrared_cache_size = DP->main_data_.grab_infrared_cache.GetDataSize();
            const int matrix_cache_size = DP->main_data_.matrix_cache.GetDataSize();
            LOG(INFO) << "[PerceptionDetectWait] " << name
                      << " send_attempt=" << attempt << "/" << kMaxSendAttempts
                      << " poll=" << poll_count << "/" << polls
                      << " rgb_cache=" << rgb_cache_size
                      << " infrared_cache=" << infrared_cache_size
                      << " matrix_cache=" << matrix_cache_size;
            if (ready()) {
                return true;
            }
        }
        return false;
    };

    auto request_with_retry = [&](uint32_t channel, uint32_t num, uint32_t interval,
                                  const auto& ready, const char* name) {
        for (int attempt = 1; attempt <= kMaxSendAttempts && rclcpp::ok(); ++attempt) {
            if (attempt > 1) {
                if (ready()) {
                    return true;
                }
                LOG(WARNING) << "Resend take photo " << name
                             << " channel=" << channel
                             << " attempt=" << attempt << "/" << kMaxSendAttempts
                             << " (previous request was likely dropped)";
                sleep_ms(kResendGapMs);
                if (ready()) {
                    return true;
                }
            }
            publish_take_photo(channel, num, interval);
            if (wait_until(ready, max_poll, attempt, name)) {
                return true;
            }
            LOG(WARNING) << "Take photo " << name << " channel=" << channel
                         << " got no data on attempt " << attempt << "/" << kMaxSendAttempts;
        }
        return ready();
    };

    if (is_truck_parking) {
        const int photo_num = param.multi_photo_num > 0 ? param.multi_photo_num : 5;

        DP->main_data_.grab_rgb_cache.Clear();
        const bool photos_ready = request_with_retry(
            kTakePhotoColorChannel, static_cast<uint32_t>(photo_num),
            static_cast<uint32_t>(interval_ms),
            [&] {
                return DP->main_data_.grab_rgb_cache.GetDataSize() >= photo_num;
            },
            "truck_parking");

        const int got = DP->main_data_.grab_rgb_cache.GetDataSize();
        if (!photos_ready) {
            LOG(WARNING) << "HandlePerceptionDetect: truck parking photos " << got
                         << "/" << photo_num;
        }
        if (got <= 0) {
            push_error("timeout waiting for truck parking photos");
            return;
        }

        const int emit = std::min(got, photo_num);
        for (int i = 0; i < emit; ++i) {
            ImageData frame;
            if (DP->main_data_.grab_rgb_cache.PopOldest(frame) < 0 || frame.img.empty()) {
                continue;
            }
            std::vector<uint8_t> encoded;
            if (!cv::imencode(".jpg", frame.img, encoded)) {
                LOG(ERROR) << "HandlePerceptionDetect: failed to encode truck parking frame " << i;
                continue;
            }
            auto info = make_event_info();
            info.object.format = "jpeg";
            info.object.data = encoded;
            if (param.save_img) {
                auto now = std::chrono::system_clock::now();
                auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count();
                std::string save_path = param.save_task_img_path + "/" + std::to_string(ts)
                    + "_truck_parking_" + std::to_string(i) + ".jpg";
                cv::imwrite(save_path, frame.img);
                LOG(INFO) << "Saved truck parking image to " << save_path;
            }
            response->event_info.push_back(std::move(info));
        }
        if (response->event_info.empty()) {
            push_error("failed to encode truck parking photos");
            return;
        }
        LOG(INFO) << "HandlePerceptionDetect: returning " << response->event_info.size()
                  << " truck parking images";
        return;
    }

    int total_occlusion_count = 0;
    bool got_clean_image = false;
    ImageData rgb_data;

    while (total_occlusion_count < max_poll && rclcpp::ok()) {
        DP->main_data_.grab_rgb_cache.Clear();
        DP->main_data_.grab_infrared_cache.Clear();
        DP->main_data_.matrix_cache.Clear();

        LOG(INFO) << "Take photo request attempt=" << total_occlusion_count;
        // IMPORTANT:
        // This function runs inside the node executor. Do not call spin_some here,
        // or the node is added twice and throws.
        const bool rgb_ok = request_with_retry(
            kTakePhotoColorChannel, 1, static_cast<uint32_t>(interval_ms),
            [&] { return DP->main_data_.grab_rgb_cache.GetDataSize() > 0; },
            "color");
        if (!rgb_ok) {
            LOG(ERROR) << "HandlePerceptionDetect: timeout waiting for color photo";
            break;
        }

        if (is_infrared_task) {
            // 彩色回图说明上一条已经处理完，再留一点间隔后发红外，避免被丢掉。
            sleep_ms(kChannelGapMs);
            const bool infrared_ok = request_with_retry(
                kTakePhotoInfraredChannel, 1, static_cast<uint32_t>(interval_ms),
                [&] {
                    return DP->main_data_.grab_infrared_cache.GetDataSize() > 0 &&
                           DP->main_data_.matrix_cache.GetDataSize() > 0;
                },
                "infrared");
            if (!infrared_ok) {
                LOG(ERROR) << "HandlePerceptionDetect: timeout waiting for infrared photo";
                break;
            }
        }

        DP->main_data_.grab_rgb_cache.PopNewest(rgb_data);

        bool occluded = CheckOcclusion(rgb_data.img);
        if (!occluded) {
            got_clean_image = true;
            break;
        }

        total_occlusion_count++;
        LOG(WARNING) << "HandlePerceptionDetect: occluded, re-grabbing ("
                     << total_occlusion_count << "/" << max_poll << ")";
    }

    if (!got_clean_image) {
        LOG(WARNING) << "HandlePerceptionDetect: occlusion timeout, returning latest data";
        if (rgb_data.img.empty()) {
            DP->main_data_.grab_rgb_cache.PopNewest(rgb_data);
        }
    }

    auto info = make_event_info();

    cv::Mat result_img;
    if (is_infrared_task) {
        ImageData infrared_data;
        DP->main_data_.grab_infrared_cache.PopNewest(infrared_data);
        result_img = infrared_data.img;

        ImageData matrix_data;
        if (DP->main_data_.matrix_cache.PopNewest(matrix_data) >= 0 &&
            !matrix_data.img.empty()) {
            double min_val, max_temp;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(matrix_data.img, &min_val, &max_temp, &min_loc, &max_loc);
            LOG(INFO) << "Max temperature: " << max_temp
                      << " at (" << max_loc.x << ", " << max_loc.y << ")";

            if (!result_img.empty()) {
                // Draw max-temperature ROI and marker on infrared image.
                int half_size = 50;
                int x1 = std::max(0, max_loc.x - half_size);
                int y1 = std::max(0, max_loc.y - half_size);
                int x2 = std::min(result_img.cols - 1, max_loc.x + half_size);
                int y2 = std::min(result_img.rows - 1, max_loc.y + half_size);
                cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                cv::rectangle(result_img, roi_rect, cv::Scalar(0, 255, 0), 2);
                cv::drawMarker(result_img, max_loc, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
            }

            common_interface::msg::BoundingBox2D box;
            box.id = 0;
            box.class_name = "max_temperature";
            box.score = static_cast<float>(max_temp);
            box.center.x = max_loc.x;
            box.center.y = max_loc.y;
            box.size_x = static_cast<float>(matrix_data.img.cols);
            box.size_y = static_cast<float>(matrix_data.img.rows);
            info.box.push_back(box);
        }

        LOG(INFO) << "HandlePerceptionDetect: returning infrared image";
    } else {
        result_img = rgb_data.img;
        LOG(INFO) << "HandlePerceptionDetect: returning RGB image";
    }

    if (!result_img.empty()) {
        std::vector<uint8_t> encoded;
        cv::imencode(".jpg", result_img, encoded);
        info.object.format = "jpeg";
        info.object.data = encoded;

        if (param.save_img) {
            auto now = std::chrono::system_clock::now();
            auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            std::string suffix = is_infrared_task ? "_infrared_result.jpg" : "_rgb_result.jpg";
            std::string save_path = param.save_task_img_path + "/" + std::to_string(ts) + suffix;
            cv::imwrite(save_path, result_img);
            LOG(INFO) << "Saved result image to " << save_path;
        }
    }
    response->event_info.push_back(info);
}


}
}
