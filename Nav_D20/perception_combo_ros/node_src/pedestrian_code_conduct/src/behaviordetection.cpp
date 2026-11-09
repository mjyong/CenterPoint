#include "../include/behaviordetection.h"
#include "math.h"

namespace seres {
namespace perception {

using json = nlohmann::json;
using namespace std::placeholders;

BehaviorDetection::BehaviorDetection(){

    node_ = std::make_shared<rclcpp::Node>("pedestrian_code_conduct_2d");
    // ipc_node_ = std::make_shared<ros_ipc_com::NodeBase>("pedestrian_code_conduct_2d");
}



bool BehaviorDetection::Init()
{

    // load config
    std::string currentFilePath(__FILE__);
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    // std::string json_config_path = std::string(parentPath) + "/../config/behaviordetection.json";
    const char* root_dir = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
    std::string perception_root = root_dir;
    std::string json_config_path = perception_root + "/pedestrian_code_conduct/config/behaviordetection.json";
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
        j.at("det_seg_ret_topic").get_to(param.det_seg_ret_topic);
        j.at("seg_src_h").get_to(param.seg_src_h);
        j.at("seg_src_w").get_to(param.seg_src_w);
        j.at("det_src_h").get_to(param.det_src_h);
        j.at("det_src_w").get_to(param.det_src_w);
        j.at("det_width_min").get_to(param.det_width_min);
        j.at("wh_ratio").get_to(param.wh_ratio);
        j.at("det_width_max").get_to(param.det_width_max);
        j.at("det_height_min").get_to(param.det_height_min);
        j.at("det_height_max").get_to(param.det_height_max);
        j.at("min_zebra_area").get_to(param.min_zebra_area);
        j.at("min_walkable_area").get_to(param.min_walkable_area);
        j.at("use_offline_mode").get_to(param.use_offline_mode);
        j.at("offline_mode_save_path").get_to(param.offline_mode_save_path);
        j.at("offline_img_dir").get_to(param.offline_img_dir);
        j.at("report_internal").get_to(param.report_internal);
        j.at("save_task_img_path").get_to(param.save_task_img_path);
        j.at("save_img").get_to(param.save_img);
        if (j.contains("perception_detect_action")) {
            j.at("perception_detect_action").get_to(param.perception_detect_action);
        }

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
    LOG(INFO) << "vis_flag: " << param.vis_flag;
    LOG(INFO) << "img_topic: " << param.img_topic;
    LOG(INFO) << "det_seg_ret_topic: " << param.det_seg_ret_topic;
    LOG(INFO) << "seg_src_h: " << param.seg_src_h;
    LOG(INFO) << "seg_src_w: " << param.seg_src_w;
    LOG(INFO) << "det_src_h: " << param.det_src_h;
    LOG(INFO) << "det_src_w: " << param.det_src_w;
    LOG(INFO) << "det_width_min: " << param.det_width_min;
    LOG(INFO) << "det_width_max: " << param.det_width_max;
    LOG(INFO) << "det_height_min: " << param.det_height_min;
    LOG(INFO) << "det_height_max: " << param.det_height_max;
    LOG(INFO) << "wh_ratio: " << param.wh_ratio;
    LOG(INFO) << "min_zebra_area: " << param.min_zebra_area;
    LOG(INFO) << "min_walkable_area: " << param.min_walkable_area;
    LOG(INFO) << "use_offline_mode: " << param.use_offline_mode;
    LOG(INFO) << "offline_mode_save_path: " << param.offline_mode_save_path;
    LOG(INFO) << "offline_img_dir: " << param.offline_img_dir;
    LOG(INFO) << "report_internal: " << param.report_internal;
    LOG(INFO) << "save_img: " << param.save_img;
    LOG(INFO) << "save_task_img_path: " << param.save_task_img_path;

    rclcpp::QoS qos_settings(10);
    // rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    // img_subscribe_ = node_->create_subscription<sensor_msgs::msg::CompressedImage>("/camera/color/image_raw", 10, std::bind(&BehaviorDetection::ImgCallback, this, std::placeholders::_1)); 
    unimg_subscribe_ = node_->create_subscription<sensor_msgs::msg::Image>(param.img_topic, 10, std::bind(&BehaviorDetection::UnImgCallback, this, std::placeholders::_1)); 

    
    pub_img_ = node_->create_publisher<sensor_msgs::msg::Image>("/behavior/vis_img", qos_settings);
    sub_det_seg_ = node_->create_subscription<nav_interface::msg::DetSegResults>(param.det_seg_ret_topic, qos_settings, std::bind(&BehaviorDetection::DetSegCallback, this, std::placeholders::_1)); 
    
    // IPC 
    // m_pub_ipc_results = ipc_node_->CreatePublisher<seres_ipc::nav_interface::msg::NavPedestrianStandard>("/robot/topic/Pedestrian/PedestrianStandard");

    // ros
    m_pub_ros_results = node_->create_publisher<nav_interface::msg::NavPedestrianStandard>("/robot/topic/Pedestrian/PedestrianStandard", qos_settings);

    // Action Server for cycle detection
    action_server_ = rclcpp_action::create_server<PerceptionDetectAction>(
        node_,
        param.perception_detect_action,
        std::bind(&BehaviorDetection::HandleGoal, this, _1, _2),
        std::bind(&BehaviorDetection::HandleCancel, this, _1),
        std::bind(&BehaviorDetection::HandleAccepted, this, _1));
    LOG(INFO) << "Action server created: " << param.perception_detect_action;

    return true;
}

void BehaviorDetection::UnImgCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{

    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        cv::Mat img = cv_ptr->image;
        if (img.cols != param.seg_src_w || img.rows != param.seg_src_h) {
            cv::resize(img, img, cv::Size(param.seg_src_w, param.seg_src_h));
        }
        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        img_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        LOG(INFO) << "GET IMG DATA: " << img_data.time;
    }
    catch (cv_bridge::Exception &e)
    {
       return;
    }
}
void BehaviorDetection::ImgCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{

    try
    {
        cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

        if (img.cols != param.seg_src_w || img.rows != param.seg_src_h) {
            cv::resize(img, img, cv::Size(param.seg_src_w, param.seg_src_h));
        }
        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        img_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        LOG(INFO) << "GET IMG DATA: " << img_data.time;
    }
    catch (cv_bridge::Exception &e)
    {
       return;
    }
}

void BehaviorDetection::DetSegCallback(const nav_interface::msg::DetSegResults::SharedPtr msg){
    if (!cycle_detect_enabled_.load()) {
        return;
    }

    DataPool* DP = DataPool::Instance();
    DetSegResults det_seg_results;
    det_seg_results.time = msg->time;
    det_seg_results.img_name = msg->img_name;

    float coord_scale_x = static_cast<float>(param.seg_src_w) / static_cast<float>(param.det_src_w);
    float coord_scale_y = static_cast<float>(param.seg_src_h) / static_cast<float>(param.det_src_h);

    for(const auto&obj:msg->dets){
        Det2D perception_object;
        perception_object.label = obj.label;
        perception_object.confidence = obj.confidence;
        if(obj.coor.size() > 3 ){
            perception_object.img_box_x = obj.coor[0] * coord_scale_x;    
            perception_object.img_box_y = obj.coor[1] * coord_scale_y;
            perception_object.img_box_w = (obj.coor[2] - obj.coor[0]) * coord_scale_x;        
            perception_object.img_box_h = (obj.coor[3] - obj.coor[1]) * coord_scale_y;        
        }else{
            LOG(WARNING) << "DetSegCallback got no det coor";
        }
                          
        det_seg_results.det_results.push_back(perception_object);
    }

    //====seg 
    for(const auto&obj:msg->segs){
        Seg2D perception_object;
        perception_object.label = obj.label;
        perception_object.confidence = obj.confidence;
        if(obj.coor.size() >= 6 ){
            for(size_t i=0;i<obj.coor.size();i+=2){
                perception_object.mask_x.push_back(obj.coor[i]   * coord_scale_x);
                perception_object.mask_y.push_back(obj.coor[i+1] * coord_scale_y); 
            }       
        }else{
            LOG(WARNING) << "DetSegCallback got no seg coor";
        }
                          
        det_seg_results.seg_results.push_back(perception_object);
    }

    DP->main_data_.detseg_cache.Push(det_seg_results);
    LOG(INFO) << "Behavior Detection got ret " << det_seg_results.time 
              << " | img_name " << det_seg_results.img_name
              << " | det size " << det_seg_results.det_results.size()
              << " | seg size " << det_seg_results.seg_results.size();

}

void BehaviorDetection::Run(){

    std::thread ros_thread = std::thread([this](){
        
        while (rclcpp::ok()){
            Process();
        }
        LOG(INFO) << "ros thread exit";
    });

    ros_thread.join();
}

void BehaviorDetection::Process()
{   
    if (!cycle_detect_enabled_.load()) {
        rclcpp::spin_some(node_);
        return;
    }

    auto start_process = std::chrono::system_clock::now();
    DataPool* DP = DataPool::Instance();

    if(DP->main_data_.detseg_cache.GetDataSize() > 0){
        
        DetSegResults det_seg_results;
        DP->main_data_.detseg_cache.PopNewest(det_seg_results);
        LOG(INFO) << std::setprecision(13) << "det_seg_results in ok time " << det_seg_results.time;
        ImageData img_data;
        cv::Mat img_show_rgb;
        if(param.use_offline_mode && det_seg_results.img_name != "none"){
            std::string img_path = param.offline_img_dir + "/" + det_seg_results.img_name;
            cv::Mat loaded = cv::imread(img_path);
            if(!loaded.empty()){
                if(loaded.cols != param.seg_src_w || loaded.rows != param.seg_src_h){
                    cv::resize(loaded, loaded, cv::Size(param.seg_src_w, param.seg_src_h));
                }
                img_show_rgb = loaded;
                LOG(INFO) << "OFFLINE: loaded image " << img_path;
            } else {
                LOG(WARNING) << "OFFLINE: failed to load " << img_path << ", fallback to black frame";
                img_show_rgb = cv::Mat(param.seg_src_h, param.seg_src_w, CV_8UC3, cv::Scalar(0, 0, 0));
            }
        } else {
            if(DP->main_data_.img_cache.GetWithinThreshold(img_data, det_seg_results.time, 100.0f, -100.0f) > -1){
                LOG(INFO) << "GET RGB IMG by timestamp match, dt=" 
                          << (det_seg_results.time - img_data.time)*1000 << "ms";
                img_show_rgb = img_data.img;
            } else if(DP->main_data_.img_cache.GetDataSize() > 0){
                LOG(INFO) << "GET RGB IMG (no timestamp match, use newest)";
                DP->main_data_.img_cache.PopNewest(img_data);
                img_show_rgb = img_data.img;
            } else {
                img_show_rgb = cv::Mat(param.seg_src_h, param.seg_src_w, CV_8UC3, cv::Scalar(0, 0, 0));
            }
        }

        int canvas_h = param.seg_src_h;
        int canvas_w = param.seg_src_w;
        float scale_x = 1.0f;
        float scale_y = 1.0f;

        // VIS MASK
        cv::Mat img_show(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat img_show_2(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat img_mask_show(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
        
        cv::Mat mask_road(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
        
        cv::Mat zebra_mask = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC1);
        cv::Mat walkable_mask = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC1);
        cv::Mat temp_mask_3 = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC1);
        for(const auto&obj:det_seg_results.seg_results){
            if(obj.label == 0){
                std::vector<cv::Point> contour;
                for(size_t i = 0; i < obj.mask_x.size(); i++) {
                    contour.push_back(cv::Point(
                        static_cast<int>(obj.mask_x[i] * scale_x),
                        static_cast<int>(obj.mask_y[i] * scale_y)));
                }
                if(contour.size() < 3) {
                    LOG(WARNING) << "zebra_crossing contour has less than 3 points, skipping. Points: " << contour.size();
                    continue;
                }
                double area = cv::contourArea(contour);
                if(area >= param.min_zebra_area) {
                    for(const auto& point : contour) {
                        cv::circle(img_show, point, 3, cv::Scalar(255, 255, 255), -1);
                        cv::circle(zebra_mask, point, 3, cv::Scalar(255), -1);
                    }
                }else{
                    if(param.vis_flag){
                        for(const auto& point : contour){
                            cv::circle(img_show, point, 2, cv::Scalar(255,255,255), -1);
                        }
                    }

                }
                

            }
            if(obj.label == 1){
                std::vector<cv::Point> contour;
                for(size_t i = 0; i < obj.mask_x.size(); i++) {
                    contour.push_back(cv::Point(
                        static_cast<int>(obj.mask_x[i] * scale_x),
                        static_cast<int>(obj.mask_y[i] * scale_y)));
                }
                if(contour.size() < 3) {
                    LOG(WARNING) << "walkable_area contour has less than 3 points, skipping. Points: " << contour.size();
                    continue;
                }
                double area = cv::contourArea(contour);

                if(area >= param.min_walkable_area) {
                    for(const auto& point : contour) {
                        cv::circle(img_show, point, 3, cv::Scalar(255, 255, 255), -1);
                        cv::circle(walkable_mask, point, 3, cv::Scalar(255), -1);
                    }
                }else{
                    if(param.vis_flag){
                        for(const auto& point : contour){
                            cv::circle(img_show, point, 2, cv::Scalar(255,255,255), -1);
                        }
                    }

                }
                

            }
        }

        std::vector<std::vector<cv::Point>> zebra_contours, walkable_contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(zebra_mask, zebra_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        cv::findContours(walkable_mask, walkable_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::vector<std::vector<cv::Point>> zebra_zone_coor;
        for(const auto& contour : zebra_contours) {
            if(contour.size() < 3) {
                    LOG(WARNING) << "zebra contour has less than 3 points, skipping. Points: " << contour.size();
                    continue;
            }
            double area = cv::contourArea(contour);
            std::vector<cv::Point> zebra_zone;
            if(area >= param.min_zebra_area) {
                for(const auto& point : contour) {
                    zebra_zone.push_back(point);
                    cv::circle(img_show, point, 6, cv::Scalar(0, 255, 255), -1);
                }
                zebra_zone_coor.push_back(zebra_zone);
            }   
        }
        LOG(INFO) << "zebra zone size: " << zebra_zone_coor.size();

        
        std::vector<std::vector<cv::Point>> walkable_zone_coor;
        for(const auto& contour : walkable_contours) {
            if(contour.size() < 3) {
                    LOG(WARNING) << "walkable contour has less than 3 points, skipping. Points: " << contour.size();
                    continue;
            }
            double area = cv::contourArea(contour);
            std::vector<cv::Point> walkable_zone;
            if(area >= param.min_walkable_area) {
                for(const auto& point : contour) {
                    walkable_zone.push_back(point);
                    cv::circle(img_show, point, 6, cv::Scalar(0, 255, 0), -1);
                }
                walkable_zone_coor.push_back(walkable_zone);
            }
            
        }
        LOG(INFO) << "walkable zone size: " << walkable_zone_coor.size();

        
        
        cv::cvtColor(zebra_mask, temp_mask_3, cv::COLOR_GRAY2BGR);
        if(param.vis_flag){
            cv::Scalar color = color3[0];
            cv::addWeighted(temp_mask_3, 0.45, img_show, 1.0, 0., img_mask_show);

        }


        std::vector<Det2D> person_data;
        bool bad_behavior = false;
        bool bad_behavior_tmp = false;
        if(det_seg_results.det_results.size() > 0){
            for(const auto&obj:det_seg_results.det_results){
                if(obj.label == 1 || obj.label == 4){
                    double wh_ratio_ = obj.img_box_w / obj.img_box_h;
                    LOG(INFO) << "DET obj.img_box_x: " << obj.img_box_x
                              << " obj.img_box_y: " << obj.img_box_y
                              << " obj.img_box_w: " << obj.img_box_w 
                              << " obj.img_box_h: " << obj.img_box_h 
                              << " wh_ratio: " << wh_ratio_;
                    
                    if(obj.img_box_w > param.det_width_min && obj.img_box_w < param.det_width_max && obj.img_box_h > param.det_height_min && obj.img_box_h < param.det_height_max && wh_ratio_ > param.wh_ratio ){
                        Det2D person;
                        person.confidence = obj.confidence;
                        person.img_box_x = obj.img_box_x;
                        person.img_box_y = obj.img_box_y;
                        person.img_box_w = obj.img_box_w;
                        person.img_box_h = obj.img_box_h;
                        person.down_center_x = obj.img_box_x + obj.img_box_w / 2;
                        person.down_center_y = obj.img_box_y + obj.img_box_h ;
                        cv::Point c_pts(person.down_center_x,person.down_center_y);
                        person_data.push_back(person);

                        LOG(INFO) << "START BEHAVIOR DETECTION: center x " << person.down_center_x << " | center y " << person.down_center_y;
                        bool in_zebra_zone = false;
                        bool in_walkable_zone = false;
                        cv::Point2f foot_pt(static_cast<float>(person.down_center_x),
                                            static_cast<float>(person.down_center_y));
                        for(const auto&contour:zebra_zone_coor){
                            double in_polygon = cv::pointPolygonTest(contour, foot_pt, false);
                            if(in_polygon >= 0){
                                LOG(INFO) << "person in zebra_crossing";
                                in_zebra_zone = true;
                                break;
                            }
                        }
                        if(!in_zebra_zone){
                            LOG(INFO) << "person not in zebra_crossing";
                        }
                        for(const auto&contour:walkable_zone_coor){
                            double in_polygon = cv::pointPolygonTest(contour, foot_pt, false);
                            if(in_polygon >= 0){
                                LOG(INFO) << "person in walkable_area";
                                in_walkable_zone = true;
                                break;
                            }
                        }
                        if(!in_walkable_zone){
                            LOG(INFO) << "person not in walkable_area";
                        }
                        if(!in_zebra_zone && !in_walkable_zone){
                            LOG(WARNING) << "BAD BEHAVIOR detected";
                            bad_behavior = true;
                            bad_behavior_tmp = true;
                        }else{
                            bad_behavior_tmp = false;
                        }


                        
                        cv::Scalar behavior_color = cv::Scalar(255, 255 , 255);
                        if(!in_zebra_zone && !in_walkable_zone){
                            behavior_color = cv::Scalar(0, 255 , 0);
                        }else if(in_zebra_zone){
                            behavior_color = cv::Scalar(0, 255 , 255);
                        }else if(in_walkable_zone){
                            behavior_color = cv::Scalar(0, 0 , 255);
                        }
                        if(param.vis_flag){
                            cv::Rect boundingBox(person.img_box_x,person.img_box_y,person.img_box_w,person.img_box_h);
                            cv::rectangle(img_mask_show, boundingBox, behavior_color, 3);
                            cv::circle(img_mask_show, c_pts, 6, cv::Scalar(0, 0, 255), -1); 
                            cv::Point up_pts(person.img_box_x ,person.img_box_y + 20);
                            cv::String text1 = "w: " + std::to_string(static_cast<int>(person.img_box_w));
                            cv::putText(img_mask_show, text1, up_pts, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                            cv::Point up_pts2(person.img_box_x + 100,person.img_box_y + 20);
                            cv::String text2 = "h: " + std::to_string(static_cast<int>(person.img_box_h));
                            cv::putText(img_mask_show, text2, up_pts2, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                            if(bad_behavior_tmp){
                                cv::String text = "bad_behavior";
                                cv::putText(img_mask_show, text, c_pts, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                            }
                        }
                    }else{
                        if(param.vis_flag){
                            cv::Rect boundingBox(obj.img_box_x,obj.img_box_y,obj.img_box_w,obj.img_box_h);
                            cv::rectangle(img_mask_show, boundingBox, cv::Scalar(255 ,255 , 255), 3);
                            cv::Point up_pts(obj.img_box_x,obj.img_box_y+20);
                            cv::String text1 = "w: " + std::to_string(static_cast<int>(obj.img_box_w));
                            cv::putText(img_mask_show, text1, up_pts, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                            cv::Point up_pts2(obj.img_box_x + 100,obj.img_box_y+20);
                            cv::String text2 = "h: " + std::to_string(static_cast<int>(obj.img_box_h));
                            cv::putText(img_mask_show, text2, up_pts2, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                        }  
                    }

                }

            }

        }


        
        
        cv::Mat img_show_final;
        cv::addWeighted(img_mask_show, 1.0, img_show_rgb, 0.6, 0., img_show_final);
    
        

        BehaviorData bad_behavior_data;
        bad_behavior_data.time = det_seg_results.time;
        bad_behavior_data.is_bad_behavior = bad_behavior;
        bad_behavior_data.img = img_show_final;
        DP->main_data_.behavior_cache.Push(bad_behavior_data);

        if(DP->main_data_.behavior_cache.GetDataSize() > 0){

            
            BehaviorData behavior_results;
            DP->main_data_.behavior_cache.PopNewest(behavior_results);
            nav_interface::msg::NavPedestrianStandard ros_msg;

            ros_msg.header.timestamp_ms= behavior_results.time * 1000;
            ros_msg.header.frame_id = "livox_frame";
            
            ros_msg.is_specification = behavior_results.is_bad_behavior;

            ros_msg.height = static_cast<uint32_t>(img_show_final.rows);
            ros_msg.width = static_cast<uint32_t>(img_show_final.cols);
            ros_msg.encoding = "bgr8";
            ros_msg.step  = static_cast<uint32_t>(img_show_final.cols * img_show_final.elemSize());
            ros_msg.is_bigendian = 0;
            
        
            size_t data_size = img_show_final.total() * img_show_final.elemSize();
            
            ros_msg.data.resize(data_size);
            LOG(INFO) << "copy data to ipc data_size: " << data_size << " ros_msg.data.size(): " << ros_msg.data.size();

            if (img_show_final.isContinuous()) {
                std::memcpy(ros_msg.data.data(), img_show_final.data, data_size);
            } else {
                size_t offset = 0;
                for (int i = 0; i < img_show_final.rows; ++i) {
                    const uint8_t* row_data = img_show_final.ptr<uint8_t>(i);
                    size_t row_size = img_show_final.cols * img_show_final.elemSize();
                    std::memcpy(ros_msg.data.data() + offset, row_data, row_size);
                    offset += row_size;
                }
            }

            bool current_is_bad = behavior_results.is_bad_behavior;
            bool should_publish_alert = false;

            
            if (current_is_bad) {
                if (internal_count_ == 0) {
                    internal_count_ = 1;
                    should_publish_alert = true;
                    LOG(INFO) << "发现异常，启动计数: 1/" << param.report_internal;
                } 
                else if (internal_count_ > 0 && internal_count_ < param.report_internal) {
                    internal_count_++;
                    should_publish_alert = false;
                    LOG(INFO) << "异常计数累加: " << internal_count_ << "/" << param.report_internal;
                    
                    if (internal_count_ >= param.report_internal) {
                        LOG(INFO) << "达到阈值" << param.report_internal << "，清零重新开始";
                        internal_count_ = 0;
                    }
                }
            } else {
                if (internal_count_ > 0) {
                    LOG(INFO) << "检测到正常帧，持续计数: " << internal_count_;
                    internal_count_ ++;
                    if (internal_count_ >= param.report_internal) {
                        LOG(INFO) << "达到阈值" << param.report_internal << "，清零重新开始";
                        internal_count_ = 0;
                    }
                }
                should_publish_alert = false;
            }

               
            ros_msg.is_specification = should_publish_alert;

            // save img 
            if(param.save_img && should_publish_alert){
                std::string save_img_path = param.save_task_img_path + "/" + std::to_string(det_seg_results.time) + "_behave.png";
                cv::imwrite(save_img_path,img_show_final);
                LOG(INFO) << "save img to " << save_img_path;
            }

            // Action feedback: 检测到异常时发送feedback
            if (should_publish_alert) {
                PublishActionFeedback(img_show_final);
            }

            // 发布消息
            m_pub_ros_results->publish(ros_msg);


        }

        if(param.use_offline_mode){
            std::string txt_file_path = param.offline_mode_save_path + "/" + det_seg_results.img_name + ".txt";
            std::string img_file_path = param.offline_mode_save_path + "/" + det_seg_results.img_name ;
            std::ofstream outfile(txt_file_path);
            if (outfile.is_open()) {
                if(bad_behavior){
                    outfile << "1";
                }else{
                    outfile << "0";
                }
                outfile.close();
                LOG(INFO) << "Bad behavior logged to file: " << txt_file_path;
                cv::imwrite(img_file_path,img_show_final);
            } else {
                outfile.close();
                LOG(ERROR) << "Unable to create log file: " << txt_file_path;
            }
        }


        cv_bridge::CvImage cv_image;
        cv_image.image = img_show_final;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        auto ros_image_msg = cv_image.toImageMsg();
        auto t3 = std::chrono::system_clock::now();

        pub_img_->publish(*ros_image_msg);

        auto end_process_1 = std::chrono::system_clock::now();
        auto ttc_1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_process_1- start_process).count() / 1000.;    
        LOG(INFO) << "Total cost time: " << ttc_1 << " ms";

    }
    
    RetsPub(); 

    rclcpp::spin_some(node_);
    return;
    
}

void BehaviorDetection::RetsPub(){

    return;
}

rclcpp_action::GoalResponse BehaviorDetection::HandleGoal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const PerceptionDetectAction::Goal> goal)
{
    bool found = false;
    uint32_t cycle = 0;
    for (const auto & evt : goal->percept) {
        if (evt.event_id == perception_interface::msg::PerceptionEvent::PERSONNEL_BEHAVIOR_STANDARD_DETECTION) {
            found = true;
            cycle = evt.detect_cycle;
            break;
        }
    }

    if (!found) {
        LOG(WARNING) << "[ActionServer] Goal rejected: PERSONNEL_BEHAVIOR_STANDARD_DETECTION not in percept list";
        return rclcpp_action::GoalResponse::REJECT;
    }
    if (cycle == 0) {
        LOG(WARNING) << "[ActionServer] Goal rejected: detect_cycle is 0";
        return rclcpp_action::GoalResponse::REJECT;
    }

    LOG(INFO) << "[ActionServer] Goal accepted, detect_cycle=" << cycle << " ms";
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse BehaviorDetection::HandleCancel(
    const std::shared_ptr<GoalHandlePerceptionDetect> /*goal_handle*/)
{
    LOG(INFO) << "[ActionServer] Cancel requested, stopping cycle detection";
    cancel_requested_.store(true);
    cycle_detect_enabled_.store(false);
    return rclcpp_action::CancelResponse::ACCEPT;
}

void BehaviorDetection::ResetDetectionState()
{
    DataPool* DP = DataPool::Instance();
    DP->main_data_.img_cache.Clear();
    DP->main_data_.detseg_cache.Clear();
    DP->main_data_.behavior_cache.Clear();
    internal_count_ = 0;
}

void BehaviorDetection::HandleAccepted(
    const std::shared_ptr<GoalHandlePerceptionDetect> goal_handle)
{
    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        if (current_goal_handle_ && current_goal_handle_->is_active()) {
            cancel_requested_.store(true);
            cycle_detect_enabled_.store(false);
        }
    }
    if (goal_thread_.joinable()) {
        goal_thread_.join();
    }

    uint32_t cycle = 0;
    const auto goal = goal_handle->get_goal();
    for (const auto & evt : goal->percept) {
        if (evt.event_id == perception_interface::msg::PerceptionEvent::PERSONNEL_BEHAVIOR_STANDARD_DETECTION) {
            cycle = evt.detect_cycle;
            break;
        }
    }

    cancel_requested_.store(false);
    detect_cycle_ms_ = cycle;
    ResetDetectionState();

    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        current_goal_handle_ = goal_handle;
        cycle_detect_enabled_.store(true);
    }

    LOG(INFO) << "[ActionServer] Cycle detection enabled, cycle=" << detect_cycle_ms_ << " ms";
    goal_thread_ = std::thread(&BehaviorDetection::ExecuteGoal, this, goal_handle);
}

void BehaviorDetection::ExecuteGoal(
    const std::shared_ptr<GoalHandlePerceptionDetect> goal_handle)
{
    LOG(INFO) << "[ActionServer] ExecuteGoal thread started";
    auto result = std::make_shared<PerceptionDetectAction::Result>();

    while (rclcpp::ok() && cycle_detect_enabled_.load()) {
        if (goal_handle->is_canceling()) {
            cancel_requested_.store(true);
            cycle_detect_enabled_.store(false);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cycle_detect_enabled_.store(false);

    if (!goal_handle->is_active()) {
        LOG(INFO) << "[ActionServer] Goal no longer active, skip terminal callback";
    } else if (!rclcpp::ok()) {
        result->message = "Pedestrian behavior detection stopped: node shutdown";
        goal_handle->abort(result);
        LOG(INFO) << "[ActionServer] Goal aborted (shutdown)";
    } else if (goal_handle->is_canceling() || cancel_requested_.load()) {
        result->message = "Pedestrian behavior detection canceled";
        goal_handle->canceled(result);
        LOG(INFO) << "[ActionServer] Goal canceled";
    } else {
        result->message = "Pedestrian behavior detection finished";
        goal_handle->succeed(result);
        LOG(INFO) << "[ActionServer] Goal succeeded";
    }

    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        if (current_goal_handle_ == goal_handle) {
            current_goal_handle_.reset();
        }
    }
    internal_count_ = 0;
    ResetDetectionState();
    LOG(INFO) << "[ActionServer] ExecuteGoal thread finished";
}

void BehaviorDetection::PublishActionFeedback(const cv::Mat & event_img)
{
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (!current_goal_handle_ || !current_goal_handle_->is_active() ||
        current_goal_handle_->is_canceling()) {
        return;
    }

    auto feedback = std::make_shared<PerceptionDetectAction::Feedback>();

    feedback->event.event_id = perception_interface::msg::PerceptionEvent::PERSONNEL_BEHAVIOR_STANDARD_DETECTION;
    feedback->event.detect_cycle = detect_cycle_ms_;

    auto now = node_->get_clock()->now();
    feedback->stamp = now;

    if (!event_img.empty()) {
        std::vector<uchar> buf;
        std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};
        cv::imencode(".jpg", event_img, buf, encode_params);

        feedback->object.format = "jpeg";
        feedback->object.data.assign(buf.begin(), buf.end());

        auto img_stamp = builtin_interfaces::msg::Time();
        img_stamp.sec = static_cast<int32_t>(now.seconds());
        img_stamp.nanosec = static_cast<uint32_t>(now.nanoseconds() % 1000000000ULL);
        feedback->object.header.stamp = img_stamp;
        feedback->object.header.frame_id = "ptz_camera";
    }

    current_goal_handle_->publish_feedback(feedback);
    LOG(INFO) << "[ActionServer] Feedback published: PERSONNEL_BEHAVIOR_STANDARD_DETECTION";
}


}
}
