#ifndef PATCHWORKPP_H
#define PATCHWORKPP_H

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <atomic>
#include <numeric>
#include <queue>
#include <mutex>
#include <thread>

#include "kygi665_msgs/msg/inspvax.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "../include/datapool.h"
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/pca.h>
//
#include <yaml-cpp/yaml.h>
// #include "comlog/comlog.h"

// serespilot_msg
// #include "serespilot_msgs/msg/lidar_cluster.hpp"
// #include "serespilot_msgs/msg/lidar_seg.hpp"
// #include <serespilot_msgs/msg/point_cloud2.hpp>
// #include <serespilot_msgs/msg/hcinspvatzcb.hpp>
#define MARKER_Z_VALUE -2.2
#define UPRIGHT_ENOUGH 0.55
#define FLAT_ENOUGH 0.2
#define TOO_HIGH_ELEVATION 0.0
#define TOO_TILTED 1.0

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 3000

using Eigen::MatrixXd;
using Eigen::JacobiSVD;
using Eigen::VectorXd;

using namespace std;
using PointT = pcl::PointXYZI;

namespace seres {
namespace perception {


struct PointXYZ {
    double x;
    double y;
    double z;
    int idx;
    int intensity;

    PointXYZ(double _x, double _y, double _z, int _idx=-1, int _intensity=0) : x(_x), y(_y), z(_z), idx(_idx),intensity(_intensity) {}
};

struct neighbour_idx {
    int zone_idx;
    int ring_idx;
    int sector_idx;

    neighbour_idx(int _zone_idx, int _ring_idx, int _sector_idx) : zone_idx(_zone_idx), ring_idx(_ring_idx), sector_idx(_sector_idx){}
};

struct patch_center {
    double center_x;
    double center_y;

    patch_center(double _center_x, double _center_y) : center_x(_center_x), center_y(_center_y) {}
};

struct RevertCandidate 
{   
    int concentric_idx;
    int sector_idx;
    double ground_flatness;
    double line_variable;
    Eigen::VectorXd pc_mean;
    vector<PointXYZ> regionwise_ground;
    
    RevertCandidate(int _c_idx, int _s_idx, double _flatness, double _line_var, Eigen::VectorXd _pc_mean, vector<PointXYZ> _ground)
     : concentric_idx(_c_idx), sector_idx(_s_idx), ground_flatness(_flatness), line_variable(_line_var), pc_mean(_pc_mean), regionwise_ground(_ground) {}
};


struct LidarSegConfigType {

    bool verbose = true;
    bool enable_RNR = false;
    bool enable_RVPF = true ;
    bool enable_TGR = true;
    bool enable_MRT = true;

    int num_iter= 3;
    int num_lpr = 15;
    int num_min_pts = 1 ;
    int num_zones = 4;
    int num_rings_of_interest = 4;//4

    double RNR_ver_angle_thr = -15.0;
    double RNR_intensity_thr = 0.2;

    double sensor_height = 1.88;
    double th_seeds = 0.25;  // 0.3  
    double th_dist = 0.25; // 离地点高度判断 非地面 0.2 
    double obstacle_height = 0.3;
    // 相对邻接地面高出不超过该值、且整体贴地的面视为可通行台阶（覆盖约 15cm 台阶及测距/平面误差）
    double step_height = 0.22;
    double th_seeds_v = 0.4; 
    double th_dist_v = 0.4;
    double max_range  = 100.0; 
    double min_range = 0.1;   
    double uprightness_thr = 0.707; // 0.707
    double adaptive_seed_selection_margin  = -1.2;
    double intensity_thr;

    double near_cal_range = 4;
    double far_cal_range = 8;

    double near_revert_thresh = 0.2;
    double middle_revert_thresh = 0.3;
    double far_revert_thresh = 0.4;

    vector<int> num_sectors_each_zone  = {32, 32, 64, 32}; 
    vector<int> num_rings_each_zone = {4, 4, 4, 4}; // 不能改
    
    int max_flatness_storage = 1000; 
    int max_elevation_storage = 1000; 

    vector<double> elevation_thr = {0, 0, 0, 0};
    vector<double> flatness_thr  = {0, 0, 0, 0}; 


};

class WildPatchWorkpp {

public:
    typedef std::vector<vector<PointXYZ>> Ring;
    typedef std::vector<Ring> Zone;

    typedef std::vector<vector<neighbour_idx>> Ring_neighbour;
    typedef std::vector<Ring_neighbour> Zone_neighbour;

    typedef std::vector<vector<patch_center>> Ring_center;
    typedef std::vector<Ring_center> Zone_center;

    WildPatchWorkpp();
    ~WildPatchWorkpp();

    int Init(YAML::Node &config);
    void Process();
    void SpinOnce();
    int LidarProcess(const LocData loc_data,
                     const LidarDataInType &lidar_raw_data,
                     pcl::PointCloud<PointT> &full_cloud,
                     std::vector<int> &foreground_idx);

    double getHeight() { return params_.sensor_height; }
    double getTimeTaken() { return time_taken_; }

    Eigen::MatrixX3d getGround() { return toEigenCloud(cloud_ground_); }
    Eigen::MatrixX3d getNonground() { return toEigenCloud(cloud_nonground_); }
    Eigen::VectorXi getGroundIndices() { return toIndices(cloud_ground_); }
    Eigen::VectorXi getNongroundIndices() { return toIndices(cloud_nonground_); }
    
    Eigen::MatrixX3d getCenters() { return toEigenCloud(centers_); }
    Eigen::MatrixX3d getNormals() { return toEigenCloud(normals_); }
private:



    bool use_multi_lidar_ = false;
    bool use_filter_ = false;
    bool use_loc_trans_ = false;
    bool use_loc_trans_remove_ = false;
    bool use_lidar_detect_ = false;
    double ncr_thresh_ = 0.03;
    double detect_num_ = 100 ;
    double filter_nb_num_ = 10.0;
    double filter_front_range_ = 20.0;
    double filter_back_range_ = -10.0;
    double filter_side_range_ = 10.0;
    double filter_nb_radius_ = 0.3;
    double convert_front_range_ = 5.0;
    double convert_blind_side_range_ = 2.0;
    double convert_blind_front_range_ = 4.0;
    int convert_count = 0;
    int convert_count_thresh_ = 10;
    bool intensity_filter_ = false;
    std::string vis_frame_id = "/livox_frame";
    std::mutex callback_mutex_,czm_mutex,publish_mutex_,loccallback_mutex_;   
    
    double filter_ground_noise_;
    double  min_x_range_;
    double  max_x_range_;
    double  min_y_range_;
    double  max_y_range_;
    double  min_z_range_;
    double  max_z_range_;
    double car_front;
    double car_back;
    double car_left;
    double car_right;
    double top_lidar_x_offset ;
    double top_lidar_y_offset ;
    double top_lidar_yaw_offset ;
    double left_lidar_yaw_offset;
    double right_lidar_yaw_offset;
    double back_lidar_x_offset ;
    double back_lidar_y_offset ;
    double back_lidar_yaw_offset ;
    double top_lidar_dis_thresh;

    // 6DoF 标定变换参数 (来自 YAML top/front/back_x/y/z_offset, yaw/pitch/roll)
    double top_x_offset_   = 0.0;
    double top_y_offset_   = 0.0;
    double top_z_offset_   = 0.0;
    double top_yaw_        = 0.0;
    double top_pitch_      = 0.0;
    double top_roll_       = 0.0;

    double front_x_offset_ = 0.0;
    double front_y_offset_ = 0.0;
    double front_z_offset_ = 0.0;
    double front_yaw_      = 0.0;
    double front_pitch_    = 0.0;
    double front_roll_     = 0.0;

    double back_x_offset_  = 0.0;
    double back_y_offset_  = 0.0;
    double back_z_offset_  = 0.0;
    double back_yaw_       = 0.0;
    double back_pitch_     = 0.0;
    double back_roll_      = 0.0;

    
    seres::perception::LidarSegConfigType params_ ;
    time_t timer_;
    long time_taken_;

    // std::vector<double> update_flatness_[4];
    // std::vector<double> update_elevation_[4];

    std::vector<double> update_flatness_[8];
    std::vector<double> update_elevation_[8];
    // 每个近处 ring 已确认地面的高程均值，用于判断低矮台阶
    double elevation_mean_[8] = {};

    double d_;

    Eigen::VectorXd normal_;
    Eigen::VectorXd singular_values_;
    Eigen::Matrix3d cov_;
    Eigen::VectorXd pc_mean_;

    vector<double> min_ranges_;
    vector<double> sector_sizes_;
    vector<double> ring_sizes_;

    vector<Zone> ConcentricZoneModel_;
    vector<Zone> ConcentricZoneModel_ground_;
    vector<Zone> ConcentricZoneModel_nonground_;
    vector<Zone_neighbour> ConcentricZoneModel_neighbour_;
    vector<Zone_center> ConcentricZoneModel_center_;

    std::shared_ptr<vector<Zone>> ConcentricZoneModel_thread;
    int threadcount = 2;
    std::vector<vector<Zone>> czm_list ;
    std::vector<std::thread> threads;

    vector<PointXYZ> ground_pc_, non_ground_pc_;
    vector<PointXYZ> regionwise_ground_, regionwise_nonground_;

    vector<PointXYZ> cloud_ground_, cloud_nonground_,cloud_revert,cloud_nonground_new_,last_cloud_to_trans,last_cloud_in_blind;

    vector<PointXYZ> centers_, normals_;

    Eigen::MatrixX3d toEigenCloud(const vector<PointXYZ> &cloud);
    Eigen::VectorXi toIndices(const vector<PointXYZ> &cloud);

    void addCloud(vector<PointXYZ> &cloud, vector<PointXYZ> &add);
    void addCloud_(vector<PointXYZ> cloud, vector<PointXYZ> &add);
    
    void flush_patches(std::vector<Zone> &czm);

    void pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm);
    void pc2czm_thread(const Eigen::MatrixXd &src, std::vector<Zone> &czm,const size_t start,const size_t end);

    void reflected_noise_removal(Eigen::MatrixXd &cloud_in);
    
    void temporal_ground_revert(std::vector<double> ring_flatness, std::vector<seres::perception::RevertCandidate> candidates, int concentric_idx);
    
    double calc_point_to_plane_d(PointXYZ p, Eigen::VectorXd normal, double d);
    void calc_mean_stdev(std::vector<double> vec, double &mean, double &stdev);

    void update_elevation_thr();
    void update_flatness_thr();

    double xy2theta(const double &x, const double &y);

    double xy2radius(const double &x, const double &y);

    void estimate_plane(const vector<PointXYZ> &ground);

    void extract_piecewiseground(
            const int zone_idx, const vector<PointXYZ> &src,
            vector<PointXYZ> &dst,
            vector<PointXYZ> &non_ground_dst);

    void extract_initial_seeds(
            const int zone_idx, const vector<PointXYZ> &p_sorted,
            vector<PointXYZ> &init_seeds);

    void extract_initial_seeds(
            const int zone_idx, const vector<PointXYZ> &p_sorted,
            vector<PointXYZ> &init_seeds, double th_seed);
    void LLtoUTM(const double Lat, const double Long, double &UTMNorthing,
                        double &UTMEasting, std::string &UTMZone);
    void LLtoUTM(const double Lat, const double Long, double &UTMNorthing,
                        double &UTMEasting, char *UTMZone);
    char UTMLetterDesignator(double Lat);
    // void HcLocalizationCallback(const serespilot_msgs::msg::Hcinspvatzcb::SharedPtr msg);
    void LidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void LidarFrontCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void LidarBackCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void InsPvaxCallback(const kygi665_msgs::msg::INSPVAX::SharedPtr msg);
    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void TransformPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& output,
                             double x, double y, double z,
                             double yaw, double pitch, double roll);

    // void LidarRightCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg);
    // void LidarLeftCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg);
    void RetsPub();


    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscribe_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscribe_front_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscribe_back_;
    rclcpp::Subscription<kygi665_msgs::msg::INSPVAX>::SharedPtr inspvax_subscribe_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscribe_;
    // 机体 pitch，弧度，抬头为正；0 表示尚未收到姿态
    std::atomic<double> body_pitch_rad_{0.0};
    // rclcpp::Subscription<serespilot_msgs::msg::PointCloud2>::SharedPtr lidar_subscribe_,lidar_subscribe_left_,lidar_subscribe_front_,lidar_subscribe_right_,lidar_subscribe_back_;
    // rclcpp::Subscription<serespilot_msgs::msg::Hcinspvatzcb>::SharedPtr loc_subscribe_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground,pub_ground_revert,pub_ground_intensity,pub_patch,pub_ground_convert,pub_ncr;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
    
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  pub_text,pub_txt_vertical_,pub_marker_init,pub_patch_init;

    // rclcpp::Publisher<serespilot_msgs::msg::LidarSeg>::SharedPtr pub_lidar_seg_ret;

    visualization_msgs::msg::MarkerArray mark_array_init_txt;
    visualization_msgs::msg::MarkerArray mark_array_poly;
    
    

//     pcl::PointCloud<PointT> revert_pc_, reject_pc_, noise_pc_, vertical_pc_;
//     pcl::PointCloud<PointT> ground_pc_,ground_vertical_pc_,dst_vertical_show_,cloud_vertical_candidate_show_;

//     pcl::PointCloud<pcl::PointXYZINormal> normals_,normals_vertical_;

//     visualization_msgs::msg::MarkerArray markerarray_,marker_array_det,markerarray_vertical_;

//     pcl::PointCloud<PointT> regionwise_ground_, regionwise_nonground_;

    
};

}
}
#endif