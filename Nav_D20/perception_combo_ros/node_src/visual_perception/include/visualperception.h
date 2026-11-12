#ifndef SERES_PERCEPTION_VISUALPERCEPTION_H__
#define SERES_PERCEPTION_VISUALPERCEPTION_H__

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <cmath> 
#include <Eigen/Dense>
#include "../include/datapool.h"
#include "../yolov8_seg/yolov8-seg.hpp"
#include "../yolov8_seg/common.hpp"
#include "../yolov8_detect/yolov8.hpp"
#include "../yolov8_detect/common.hpp"

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

//
#include <nav_interface/msg/det_seg_results.hpp>
#include <nav_interface/msg/visual2_d.hpp>
#include <nav_interface/msg/nav_sensor_camera.hpp>


// // ipc 故障诊断
// #include <nav_interface/visual_perception_state.h>
// #include "base_node.h"

namespace seres {
namespace perception {

// 故障码枚举
constexpr static uint32_t OK = 0;
// 定位错误
constexpr static uint32_t LOC_ODOMETRY_DIVERGENCE = 1;        // 里程计残差不收敛
constexpr static uint32_t LOC_TOO_FEW_MATCHES = 2;            // 匹配点数过少
constexpr static uint32_t LOC_LOW_MATCH_SCORE = 3;            // 匹配得分底
constexpr static uint32_t LOC_MAP_MATCH_SCORE_LOW = 4;        // 地图匹配得分太低
constexpr static uint32_t LOC_MAP_MISMATCH = 5;               // 地图误匹配
constexpr static uint32_t MAP_OPTIMIZATION_DIVERGENCE = 6;    // 优化不收敛
constexpr static uint32_t MAP_TOO_FEW_MATCHES = 7;            // 匹配点数过少
constexpr static uint32_t MAP_LOW_MATCH_SCORE = 8;            // 匹配得分低
constexpr static uint32_t MAP_LOOP_CLOSURE_FAULT = 9;         // 回环异常
constexpr static uint32_t MAP_BACKEND_OPTIMIZATION_FAIL = 10; // 后段优化不收敛
                                                              // 规控错误
constexpr static uint32_t PNC_LOC_TIMEOUT = 11;               // 定位信息超时
constexpr static uint32_t PNC_LOCAL_MAP_TIMEOUT = 12;         // 局部占用栅格地图超时
constexpr static uint32_t PNC_GLOBAL_MAP_LOAD_FAIL = 13;      // 全局地图加载失败
constexpr static uint32_t PNC_GLOBAL_PLAN_FAIL = 14;          // 全局路径规划失败
constexpr static uint32_t PNC_TASK_FILE_PARSE_FAIL = 15;      // 任务文件解析失败
constexpr static uint32_t PNC_TASK_INFO_PARSE_FAIL = 16;      // 任务信息解析失败
constexpr static uint32_t PNC_GIMBAL_TIMEOUT = 17;            // 云台状态信息超时
constexpr static uint32_t PNC_POSE_DEVIATION_LARGE = 18;      // 当前位置与全局参考路径偏差过大
constexpr static uint32_t PNC_LOCAL_REF_PATH_ERROR = 19;      // 局部参考路径提取错误
constexpr static uint32_t PNC_CONTROL_COMPUTE_FAIL = 20;      // 控制指令解算异常
constexpr static uint32_t PNC_LOCAL_PLAN_SOLVER_FAIL = 21;    // 局部规划求解失败
constexpr static uint32_t PNC_LOCAL_PLAN_STOP_BY_OBS = 22;    // 局部规划停障
constexpr static uint32_t PNC_EMERGENCY_STOP = 23;            // 因障碍栅格急停
                                                              // 传感器数据异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_TIMESTAMP = 24; // 传感器时间戳异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_FREQUENCY = 25; // 数据频率异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_CONTENT = 26;   // 数据内容异常


// const std::vector<std::string> SEG_CLASS_NAMES = {
//             "road","sidewalk","zebra_crossing"
// };

const std::vector<std::string> SEG_CLASS_NAMES = {
            "zebra_crossing","walkable_area"
};

// const std::vector<std::string> SEG_CLASS_NAMES = {
//             "zebra_crossing"
// };


const std::vector<std::vector<unsigned int>> SEG_COLORS = {
    {207, 0, 0},       // Tyre - 红色
    {250, 191, 0},     // Ashtray - 棕色
    {150, 150, 150},     
};

const std::vector<std::vector<unsigned int>> SEG_MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

struct InitParameter
{
    int gpu_id = 0;
    int num_class{ 80 }; // coco 
    std::vector<std::string> class_names;
    std::vector<std::string> input_output_names;

    bool dynamic_batch{ true };
    size_t batch_size;
    int src_h, src_w; 
    int dst_h, dst_w;

    float scale{ 255.f };
    float means[3] = { 0.f, 0.f, 0.f };
    float stds[3] = { 1.f, 1.f, 1.f };

    float iou_thresh;
    float conf_thresh;

    int topk = 100;
    int seg_channels = 32;
    int seg_h = 160;
    int seg_w = 160;
    std::string save_path;

    int char_width = 11; 
    int det_info_render_width = 15;
    double font_scale = 0.6;
    bool is_show = false;
    bool is_save = false;
    std::string model_path ;
    std::string img_topic ;
    std::string infrared_topic ;

    bool use_offline_mode;
    std::string offline_img_file_path;
    bool use_ptz_camera;

    int frame_process_interval;
    bool pub_state_ipc;
    std::string ipc_state_topic;
};

class VisualPerception
{
public:
    // 构造函数
    VisualPerception();
    bool Init();
    void Process();
    void Run();

private:
    rclcpp::Node::SharedPtr node_;
    // std::shared_ptr<ros_ipc_com::NodeBase> ipc_node_;

    // std::shared_ptr<ros_ipc_com::PublisherBase<seres_ipc::nav_interface::msg::VisualPerceptionState>> m_pub_ipc_state;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_subscribe_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr unimg_subscribe_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr ptz_img_subscribe_;
    rclcpp::Subscription<nav_interface::msg::NavSensorCamera>::SharedPtr ptz_infrared_subscribe_;


    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_,ptz_img_pub_,ptz_infrared_pub_,offline_img_pub_;
    rclcpp::Publisher<nav_interface::msg::DetSegResults>::SharedPtr pub_det_seg_;
    
    void ImgCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void UnImgCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void PTZImgCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void PTZInfraredCallback(const nav_interface::msg::NavSensorCamera::SharedPtr msg);
    
    void RetsPub();
  
private:
    std::mutex callback_mutex_;
    std::mutex publish_mutex_;
    // static std::mutex cloud_mutex;

    double fx = 606.757080078125;
    double fy = 606.7099609375;
    double cx = 327.0721130371094;
    double cy = 250.8756866455078;

    // 坐标转换 先绕y轴转90度 再 绕x轴转-90度 最后再绕y轴转-PI/8度
    float yaw_rad = -M_PI / 2; // x 90度
    float pitch_rad = M_PI / 2 ; // y -90度
    float roll_rad = -M_PI/2   ; // x 45度
    float pitch_rad_2 = -M_PI / 6 ; // 
    // 标定参数
    float x_offset = 0.366;
    float y_offset = 0;
    float z_offset = -0.071;

    Eigen::Matrix3f R;
    Eigen::Vector3f T;

    float time_thresh = 2000.0; // 离线调试没有办法 先把数字放大
    float IOU_thresh = 0.5; // 离线调试没有办法 先把数字放大
    float IOU_face_thresh = 0.02; // 离线调试没有办法 先把数字放大

    bool vis_flag = true;
    std::string vis_frame_id = "dog";

    InitParameter param;
    // std::shared_ptr<YOLOv8Seg> yolo;
    std::shared_ptr<YOLOv8_seg> yolov8_seg_;

    std::shared_ptr<YOLOv8> yolov8_;
    int frame_count = 0;
    double offline_img_time = 0.0;
};


} // namespace perception
} // namespace seres

#endif
