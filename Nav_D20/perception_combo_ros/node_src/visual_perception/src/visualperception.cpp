#include "../include/visualperception.h"
#include "../include/logging.h"
#include "math.h"

namespace seres {
namespace perception {

using json = nlohmann::json;

VisualPerception::VisualPerception(){
  node_ = std::make_shared<rclcpp::Node>("visual_perception_2d");
//   ipc_node_ = std::make_shared<ros_ipc_com::NodeBase>("visual_perception_2d");
}


// void setParameters(InitParameter& initParameters)
// {
// 	// initParameters.class_names = utils::dataSets::coco80;
// 	// initParameters.num_class = 80; // for coco
// 	initParameters.class_names = seres_seg_3cls;
// 	initParameters.num_class = 3; // for voc2012
// 	initParameters.batch_size = 1;
// 	initParameters.dst_h = 640;
// 	initParameters.dst_w = 640;
// 	initParameters.input_output_names = { "images",  "output0" ,"output1"};
// 	initParameters.conf_thresh = 0.5f;
// 	initParameters.iou_thresh = 0.5f;
// 	initParameters.save_path = "~/";
// }


bool VisualPerception::Init()
{

    
    // setParameters(param);
    // load config
    std::string currentFilePath(__FILE__);
    std::filesystem::path currentPath(currentFilePath);
    std::filesystem::path parentPath = currentPath.parent_path();
    // std::string json_config_path = std::string(parentPath) + "/../config/visualperception.json";
    const char* root_dir = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
    std::string perception_root = root_dir;
    std::string json_config_path = perception_root + "/visual_perception/config/visualperception.json";
    std::ifstream ifs(json_config_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + json_config_path);
        return false;
    }
    json j; 
    ifs >> j;
    std::string det_engine_file_path_;
    try {
        j.at("gpu_id").get_to(param.gpu_id);
        j.at("img_topic").get_to(param.img_topic);
        j.at("infrared_topic").get_to(param.infrared_topic);
        j.at("seg_batch_size").get_to(param.batch_size);
        j.at("seg_model_path").get_to(param.model_path);
        j.at("seg_is_show").get_to(param.is_show);
        j.at("seg_is_save").get_to(param.is_save);
        j.at("seg_src_h").get_to(param.src_h);
        j.at("seg_src_w").get_to(param.src_w);
        j.at("seg_num_cls").get_to(param.num_class);
        j.at("seg_dst_h").get_to(param.dst_h);
        j.at("seg_dst_w").get_to(param.dst_w);
        j.at("seg_conf_thresh").get_to(param.conf_thresh);
        j.at("seg_iou_thresh").get_to(param.iou_thresh);
        j.at("seg_dst_w").get_to(param.dst_w);
        j.at("det_model_path").get_to(det_engine_file_path_);
        j.at("use_offline_mode").get_to(param.use_offline_mode);
        j.at("offline_img_file_path").get_to(param.offline_img_file_path);
        j.at("use_ptz_camera").get_to(param.use_ptz_camera);
        j.at("frame_process_interval").get_to(param.frame_process_interval);
        j.at("pub_state_ipc").get_to(param.pub_state_ipc);
        j.at("ipc_state_topic").get_to(param.ipc_state_topic);
        
        
        

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
    LOG(INFO) << "GPU ID: " << param.gpu_id;
    LOG(INFO) << "Image Topic: " << param.img_topic;
    LOG(INFO) << "Model Path: " << param.model_path;
    LOG(INFO) << "Batch Size: " << param.batch_size;
    LOG(INFO) << "Model Path: " << param.model_path;
    LOG(INFO) << "Show: " << (param.is_show ? "true" : "false");
    LOG(INFO) << "Save: " << (param.is_save ? "true" : "false");
    LOG(INFO) << "Source Height: " << param.src_h;
    LOG(INFO) << "Source Width: " << param.src_w;
    LOG(INFO) << "Number of Classes: " << param.num_class;
    LOG(INFO) << "Destination Height: " << param.dst_h;
    LOG(INFO) << "Destination Width: " << param.dst_w;
    LOG(INFO) << "Confidence Threshold: " << param.conf_thresh;
    LOG(INFO) << "IOU Threshold: " << param.iou_thresh;
    LOG(INFO) << "use offline mode: " << param.use_offline_mode;
    LOG(INFO) << "offline_img_file_path: " << param.offline_img_file_path;
    LOG(INFO) << "use_ptz_camera: " << param.use_ptz_camera;
    LOG(INFO) << " param.frame_process_interval " << param.frame_process_interval;
    LOG(INFO) << " pub_state_ipc " << param.pub_state_ipc;
    LOG(INFO) << " ipc_state_topic " << param.ipc_state_topic;

    rclcpp::QoS qos_settings_default(10);
    rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    // rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::KeepLast(10))
    // .reliability(rclcpp::ReliabilityPolicy::Reliable);  
    // img_subscribe_ = node_->create_subscription<sensor_msgs::msg::CompressedImage>("/camera/color/image_raw", 10, std::bind(&VisualPerception::ImgCallback, this, std::placeholders::_1)); 
    if(param.use_ptz_camera){
        // 云台 RGB：sensor_msgs/Image，QoS = BestEffort + KeepLast(10) + Volatile
        ptz_img_subscribe_ = node_->create_subscription<sensor_msgs::msg::Image>(
            param.img_topic, qos_settings,
            std::bind(&VisualPerception::PTZImgCallback, this, std::placeholders::_1));

        ptz_img_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/perception/ptz_ori_image", qos_settings_default);

        // ptz_infrared_subscribe_ = node_->create_subscription<nav_interface::msg::NavSensorCamera>(param.infrared_topic, 10, std::bind(&VisualPerception::PTZInfraredCallback, this, std::placeholders::_1));

        ptz_infrared_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/perception/ptz_infrared_image", qos_settings_default);
    }

    if(!param.use_offline_mode && !param.use_ptz_camera){
        unimg_subscribe_ = node_->create_subscription<sensor_msgs::msg::Image>(param.img_topic, 10, std::bind(&VisualPerception::UnImgCallback, this, std::placeholders::_1));

    }else if(param.use_offline_mode){

        offline_img_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/perception/offline_image", qos_settings_default);
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
            std::cout << "提取的文件名: " << filename_split << std::endl;

            img_data.img_name = filename_split;
            DP->main_data_.img_cache.Push(img_data);
            std::cout << " --------GET IMG DATA : " << filename << std::endl;
            img_count += 0.1;


        }
        std::cout << " TOTAL IMG NUM " << img_count << std::endl;

    }
    // unimg_subscribe_ = node_->create_subscription<sensor_msgs::msg::Image>(param.img_topic, 10, std::bind(&VisualPerception::UnImgCallback, this, std::placeholders::_1)); 
    pub_img_ = node_->create_publisher<sensor_msgs::msg::Image>("/camera/seg_det_result", qos_settings_default);
    pub_det_seg_ = node_->create_publisher<nav_interface::msg::DetSegResults>("/perception/seg_det_result", qos_settings_default);

    // YOLOv8Seg 
    
    yolov8_seg_.reset(new YOLOv8_seg(param.model_path));
    yolov8_seg_->make_pipe(true);
    // yolo = std::make_shared<YOLOv8Seg>(param); 
	// std::vector<unsigned char> trt_file = utils::loadModel(std::string(param.model_path));
	// if (trt_file.empty())
	// {
	// 	std::cout << "trt_file is empty!" << std::endl;
	// 	return -1;
	// }
	// if (!yolo->init(trt_file))
	// {
	// 	std::cout << "initEngine() ocur errors!" << std::endl;
	// 	return -1;
	// }
    // yolo->check();
    // std::cout << "check()" << std::endl;
    // ========det init
    // std::string engine_file_path_ = "/home/huanghanlin/hhl_work/seres_work/dog_follow/D10_related/ros2_node/visual_perception/model/D10_super_factory_road_data_3cls_detect_yolov8s_640x640_new.trt";
    yolov8_.reset(new YOLOv8(det_engine_file_path_));
    // yolov8_ = std::make_shared<YOLOv8>(engine_file_path_);
    // std::cout << "make_pipe()" << std::endl;
    yolov8_->make_pipe(true);



    // IPC Publish
    // m_pub_ipc_state = ipc_node_->CreatePublisher<seres_ipc::nav_interface::msg::VisualPerceptionState>(param.ipc_state_topic);


    return true;
}

void VisualPerception::Run(){


    // std::thread ipc_thread = std::thread([this](){

    //     ipc_node_->run();
    //     std::cout << "ipc thread exit" << std::endl;
    // });

    std::thread ros_thread = std::thread([this](){
        
        while (rclcpp::ok()){
            Process();
        }
        std::cout << "ros thread exit" << std::endl;
    });

    // ipc_thread.join();
    ros_thread.join();

}

void VisualPerception::PTZInfraredCallback(const nav_interface::msg::NavSensorCamera::SharedPtr msg)
{
    std::cout << " IN PTZInfraredCallback " << std::endl;
    try
    {
        auto image_msg = std::make_shared<sensor_msgs::msg::Image>() ;
        
        // image_msg->header = msg->header;
        double time_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        image_msg->height = msg->height;
        image_msg->width = msg->width;
        image_msg->encoding = msg->encoding;
        image_msg->data = msg->data;
        image_msg->step = msg->width * 3;
        image_msg->is_bigendian = false;
            std::cout << "Received image - Encoding: " << msg->encoding 
              << ", Size: " << msg->width << "x" << msg->height
              << ", Data size: " << msg->data.size() << " bytes" << std::endl;
        // 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        // cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

        // 访问 cv::Mat 图像
        cv::Mat img = cv_ptr->image;
        // 在此处处理图像
        // std::unique_lock<std::mutex> locker(callback_mutex_);
        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        // img_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        img_data.time =  time_stamp;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        std::cout << " --------GET PTZ Infrared DATA : " << img_data.time  << std::endl;

        cv_bridge::CvImage cv_image;
        cv_image.image = img;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;

        auto ros_image_msg = cv_image.toImageMsg();

        ptz_infrared_pub_->publish(*ros_image_msg);
    }
    catch (cv_bridge::Exception &e)
    {
        std::cout << " --------GET PTZ Infrared ERR : " << std::endl;
       return;
    }
}
void VisualPerception::PTZImgCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    std::cout << " IN PTZImgCallback " << std::endl;

    frame_count += 1;
    if(frame_count < param.frame_process_interval){
        std::cout << " --------------- FRAME FILTER  ------------ "  << frame_count << std::endl;
        return;
    }
    frame_count = 0;
    try
    {
        double time_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::cout << "Received image - Encoding: " << msg->encoding
                  << ", Size: " << msg->width << "x" << msg->height
                  << ", Data size: " << msg->data.size() << " bytes" << std::endl;

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        img_data.time = time_stamp;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        std::cout << " --------GET PTZ IMG DATA : " << img_data.time  << std::endl;

        // 转发给 pedestrian_code_conduct 等下游节点
        cv_bridge::CvImage cv_image;
        cv_image.header = msg->header;
        cv_image.image = img;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        auto ros_image_msg = cv_image.toImageMsg();
        ptz_img_pub_->publish(*ros_image_msg);
    }
    catch (cv_bridge::Exception &e)
    {
        std::cout << " --------GET PTZ IMG ERR : " << e.what() << std::endl;
       return;
    }
}
void VisualPerception::UnImgCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{

    
    try
    {
        // 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

        // 访问 cv::Mat 图像
        cv::Mat img = cv_ptr->image;
        // 在此处处理图像
        // std::unique_lock<std::mutex> locker(callback_mutex_);
        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        img_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        std::cout << " --------GET IMG DATA : " << img_data.time  << std::endl;
    }
    catch (cv_bridge::Exception &e)
    {
       return;
    }
}
void VisualPerception::ImgCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{

    try
    {
        // 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
        // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

        // 访问 cv::Mat 图像
        // cv::Mat img = cv_ptr->image;
        // 在此处处理图像
        // std::unique_lock<std::mutex> locker(callback_mutex_);
        DataPool* DP = DataPool::Instance();
        ImageData img_data;
        img_data.time = msg->header.stamp.sec + msg->header.stamp.nanosec*1e-9;
        img_data.img = img;
        DP->main_data_.img_cache.Push(img_data);
        std::cout << " --------GET IMG DATA : " << img_data.time  << std::endl;
    }
    catch (cv_bridge::Exception &e)
    {
       return;
    }
}


void VisualPerception::Process()
{   
    auto start_process = std::chrono::system_clock::now();
    DataPool* DP = DataPool::Instance();
    // 深度相机点云处理
    
    if(DP->main_data_.img_cache.GetDataSize() > 0){
        std::cout << "DP->main_data_.img_cache.GetDataSize() " <<DP->main_data_.img_cache.GetDataSize() <<std::endl;
        // 获取数据
        ImageData img_data;
        DP->main_data_.img_cache.PopOldest(img_data); //先拿最旧的数据处理
        std::cout << "====== img_cache in ok time "  << img_data.time << std::endl;
        if(img_data.img_name != "none"){
            std::cout << " img path " << img_data.img_name <<std::endl;
        }
        if(param.use_offline_mode){
                // sleep(0.1);
                // std::this_thread::sleep_for(std::chrono::milliseconds(50));

    
                cv_bridge::CvImage cv_image;
                cv_image.image = img_data.img;
                cv_image.encoding = sensor_msgs::image_encodings::BGR8;

                auto ros_image_msg = cv_image.toImageMsg();
                ros_image_msg->header.stamp.sec = trunc(offline_img_time);
                ros_image_msg->header.stamp.nanosec = double(offline_img_time - trunc(offline_img_time))*1000;
                offline_img_time +=0.1;
                offline_img_pub_->publish(*ros_image_msg);

        }
        // 
        std::vector<cv::Mat> imgs_batch;
        imgs_batch.reserve(param.batch_size);
        imgs_batch.emplace_back(img_data.img.clone());
            
        auto end_process_1 = std::chrono::system_clock::now();
        auto ttc_1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_process_1- start_process).count() / 1000.;    
        std::cout << "step 1 time: " << ttc_1 << " ms" << std::endl;

        int delay_time = 0;
        int batchi = 0;
        // task(yolo, param, imgs_batch, delay_time, batchi);
        // std::cout << " 11111111 " << std::endl;
        ProcessResult results;
        if(yolov8_seg_){
            // yolo->copy(imgs_batch);
            // // std::cout << " 2222222222 " << std::endl;
            // utils::DeviceTimer d_t1; yolo->preprocess(imgs_batch);  float t1 = d_t1.getUsedTime();
            // utils::DeviceTimer d_t2; yolo->infer();				  float t2 = d_t2.getUsedTime();
            // utils::DeviceTimer d_t3; yolo->postprocess(imgs_batch); float t3 = d_t3.getUsedTime();
            // float avg_times[3] = { t1 / param.batch_size, t2 / param.batch_size, t3 / param.batch_size };
            // std::cout << "preprocess time = " << avg_times[0] << "; "
            //     "infer time = " << avg_times[1] << "; "
            //     "postprocess time = " << avg_times[2] << std::endl;
            // yolo->showAndSave(param.class_names, delay_time, imgs_batch,results);
            // yolo->reset();
            std::vector<Object> objs_;
            cv::Mat  img_res_, image_;
            cv::Size targetSize(param.dst_w, param.dst_h);
            objs_.clear();
            yolov8_seg_->copy_from_Mat(img_data.img.clone(), targetSize);
            yolov8_seg_->infer();
            yolov8_seg_->postprocess(objs_, param.conf_thresh, param.iou_thresh, param.topk, param.seg_channels, param.seg_h, param.seg_w);
            yolov8_seg_->draw_objects(img_data.img.clone(), img_res_, objs_, SEG_CLASS_NAMES, SEG_COLORS, SEG_MASK_COLORS,results);

        }

        // pub result
        // 访问结果 目前单batch
        cv::Mat overlay_img ;
        DetSegResults det2d_seg2d_results;
        if(param.use_offline_mode){
            det2d_seg2d_results.img_name = img_data.img_name;
        }else{
            det2d_seg2d_results.img_name = "none";
        }
        
        det2d_seg2d_results.time = img_data.time;
        // for (const auto& result : results) {// batch size result now is 1
            // 获取叠加图像
            overlay_img = results.overlay_image;

            for (const auto& detection : results.detections) {
                std::cout << "Detected: " << detection.class_name 
                        << " Confidence: " << detection.confidence
                        << " BBox: " << detection.bbox 
                        // << " Mask: " << detection.mask
                        << std::endl;
                // 单类别的分割结果
                Seg2D seg2d;
                seg2d.label = detection.class_id;
                seg2d.confidence = detection.confidence;
                seg2d.mask_x.clear();
                seg2d.mask_y.clear();
                // 可以使用 detection.mask 获取分割mask的轮廓边界点
                if (!detection.mask.empty()) {
                    // 将掩码转换为二值图像
                    cv::Mat binary_mask;
                    cv::threshold(detection.mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
                    binary_mask.convertTo(binary_mask, CV_8UC1);

                    // 方法1: 高斯模糊平滑边缘
                    cv::Mat smoothed_mask;
                    cv::GaussianBlur(binary_mask, smoothed_mask, cv::Size(5, 5), 1.0, 1.0);
                    
                    // 重新阈值化
                    cv::threshold(smoothed_mask, smoothed_mask, 127, 255, cv::THRESH_BINARY);
                    // 查找轮廓
                    std::vector<std::vector<cv::Point>> contours;
                    std::vector<cv::Vec4i> hierarchy;
                    cv::findContours(smoothed_mask, 
                                     contours, 
                                     hierarchy, 
                                     cv::RETR_EXTERNAL, 
                                     cv::CHAIN_APPROX_NONE,
                                     cv::Point(detection.bbox.x, detection.bbox.y));
                    
                    // 处理每个轮廓
                    std::vector<std::vector<cv::Point>> contours_down;
                    for (size_t i = 0; i < contours.size(); i++) {
                        std::vector<cv::Point> contours_i = contours[i];
                        if (contours_i.size() > 2) { // 确保轮廓有足够的点
                            // // 简化轮廓（可选，减少点数）不减少点数
                            // std::vector<cv::Point> approx_contour;
                            // // double epsilon = 0.001 * cv::arcLength(contours_i, true);
                            // cv::approxPolyDP(contours_i, approx_contour, 0.2, true);

                            std::cout << "Contour " << i << " has " << contours_i.size() << " points:" << std::endl;
                            // 输出边界点坐标
                            // for (auto& point : contours_i) {
                                
                            //     point.x = point.x  + detection.bbox.x;
                            //     point.y = point.y  + detection.bbox.y;
                            //     // std::cout << "Point: (" << point.x << ", " << point.y << ")" << std::endl;
                            // }
                            
                            // 可选：在图像上绘制边界点（用于可视化）
                            for (const auto& point : contours_i) {
                                seg2d.mask_x.emplace_back(point.x);
                                seg2d.mask_y.emplace_back(point.y);
                                cv::circle(overlay_img, point, 3, cv::Scalar(0, 255, 0), -1); // 绿色点
                            }
                            
                            // for vis
                            contours_down.emplace_back(contours_i);
                            
                        }
                    }
                    cv::drawContours(overlay_img, contours_down, -1, cv::Scalar(0, 0, 255), 2);
                }
                det2d_seg2d_results.seg_results.emplace_back(seg2d);
            }

        
        // }
        auto end_process_2 = std::chrono::system_clock::now();
        auto ttc_2 = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_process_2 - start_process).count() / 1000.;    
        std::cout << "step 2 time: " << ttc_2 << " ms" << std::endl;
        // =============== det 
        cv::Size size_ = cv::Size{640, 640};
        cv::Mat img_res_;
        std::vector<Object> objs_;
        yolov8_->copy_from_Mat(img_data.img.clone(), size_);
        auto start = std::chrono::system_clock::now();
        yolov8_->infer();
        
        // 默认版本
        yolov8_->postprocess(objs_);
        yolov8_->draw_objects(overlay_img, img_res_, objs_, CLASS_NAMES, COLORS);
        auto end = std::chrono::system_clock::now();

        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;    
        std::cout << "det Inference time: " << tc << " ms" << std::endl;

        // 解析分割和检测结果
        for(const auto& obj : objs_){
            std::cout << "DET label: " << obj.label 
                     << " prob: " << obj.prob 
                     << " box x " << obj.rect.x 
                     << " box y " << obj.rect.y 
                     << " box width " << obj.rect.width
                     << " box height " << obj.rect.height
                     << std::endl;

            Det2D det2d;
            det2d.label = obj.label;
            det2d.confidence = obj.prob;
            det2d.img_box_x = obj.rect.x;
            det2d.img_box_y = obj.rect.y;
            det2d.img_box_w = obj.rect.width;
            det2d.img_box_h = obj.rect.height;
            det2d_seg2d_results.det_results.push_back(det2d);
        }


        DP->main_data_.detseg_cache.Push(det2d_seg2d_results);

        // cv_bridge::CvImage cv_image;
        // cv_image.image = img_res_;
        // cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        // auto ros_image_msg = cv_image.toImageMsg();
        // pub_img_->publish(*ros_image_msg);

        // auto end_process_3 = std::chrono::system_clock::now();
        // auto ttc_3 = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_process_3 - end).count() / 1000.;    
        // std::cout << "step 3 time: " << ttc_3 << " ms" << std::endl;

        // 详细计时
        auto t1 = std::chrono::system_clock::now();
        cv_bridge::CvImage cv_image;
        cv_image.image = img_res_;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        cv_image.header.stamp = node_->now();
        auto t2 = std::chrono::system_clock::now();

        auto ros_image_msg = cv_image.toImageMsg();
        auto t3 = std::chrono::system_clock::now();

        pub_img_->publish(*ros_image_msg);
        auto t4 = std::chrono::system_clock::now();

        auto time_cvimage = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.;
        auto time_toImageMsg = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.;
        auto time_publish = (double)std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.;

        std::cout << "CVImage creation: " << time_cvimage << " ms" << std::endl;
        std::cout << "toImageMsg: " << time_toImageMsg << " ms" << std::endl;
        std::cout << "publish: " << time_publish << " ms" << std::endl;

        auto end_process = std::chrono::system_clock::now();
        auto ttc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_process- start_process).count() / 1000.;    
        std::cout << "Total Inference time: " << ttc << " ms" << std::endl;
    }
    // else if(param.pub_state_ipc){


    //     seres_ipc::nav_interface::msg::VisualPerceptionState ipc_msg;

    //     ipc_msg.state = SENSOR_DATA_ANOMALY_TIMESTAMP;

    //     m_pub_ipc_state->send_message(ipc_msg);

    //     ipc_msg.state = SENSOR_DATA_ANOMALY_FREQUENCY;

    //     m_pub_ipc_state->send_message(ipc_msg);

    //     ipc_msg.state = SENSOR_DATA_ANOMALY_CONTENT;

    //     m_pub_ipc_state->send_message(ipc_msg);

    //     std::cout << " NO DATA , STATE ERROR " <<std::endl;
        
    // }
    
    RetsPub(); 
    rclcpp::spin_some(node_);
    return;
    
}

void VisualPerception::RetsPub(){
    // std::unique_lock<std::mutex> locker(publish_mutex_);
    DataPool* DP = DataPool::Instance();
    DetSegResults det_seg_results;
    nav_interface::msg::DetSegResults det_seg_ouput;

    if(DP->main_data_.detseg_cache.GetDataSize() > 0){

        if(DP->main_data_.detseg_cache.PopOldest(det_seg_results) > -1){
            // if(DP->main_data_.rs2_rets_cache.PopNewest(rs2_dets) > -1){
            
            det_seg_ouput.time = det_seg_results.time;
            if(param.use_offline_mode){
                det_seg_ouput.img_name = det_seg_results.img_name;
            }else{
                det_seg_ouput.img_name = "none";
            }
            std::cout << std::setprecision(13) << " -------DET SEG  RESULT : " << det_seg_ouput.time << std::endl;
            for(auto &obj:det_seg_results.det_results){
                // std::cout << " ----- PUB rs2_dets " << obj.x
                //         << " | y " << obj.y
                //         << " | s " << obj.source 
                //         << std::endl;
                nav_interface::msg::Visual2D perception_object;
                perception_object.label = obj.label;
                perception_object.confidence = obj.confidence;
                perception_object.class_name   = CLASS_NAMES[obj.label];
                perception_object.task   = "detection";
                perception_object.coor.push_back(obj.img_box_x);
                perception_object.coor.push_back(obj.img_box_y);
                perception_object.coor.push_back(obj.img_box_x+obj.img_box_w);
                perception_object.coor.push_back(obj.img_box_y+obj.img_box_h);
                det_seg_ouput.dets.push_back(perception_object);
            }
            for(auto &seg:det_seg_results.seg_results){
                // std::cout << " ----- PUB rs2_dets " << obj.x
                //         << " | y " << obj.y
                //         << " | s " << obj.source 
                //         << std::endl;
                nav_interface::msg::Visual2D perception_seg;
                perception_seg.label = seg.label;
                perception_seg.confidence = seg.confidence;
                perception_seg.class_name = SEG_CLASS_NAMES[seg.label];
                perception_seg.task   = "segmentation";
                for(int i=0;i<seg.mask_x.size();i++){
                    perception_seg.coor.push_back(seg.mask_x[i]);
                    perception_seg.coor.push_back(seg.mask_y[i]);
                }

                det_seg_ouput.segs.push_back(perception_seg);
            }

            pub_det_seg_->publish(det_seg_ouput);

        }

    }

    return;
}



}
}
