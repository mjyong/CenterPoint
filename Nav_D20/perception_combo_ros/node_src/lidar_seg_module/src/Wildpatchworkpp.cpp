#include "../include/Wildpatchworkpp.h"
#include <pcl/console/time.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
// #include <tf2/LinearMath/Quaternion.h>

namespace seres {
namespace perception {

WildPatchWorkpp::WildPatchWorkpp() {
    node_ = std::make_shared<rclcpp::Node>("lidar_seg_module");
    ConcentricZoneModel_thread = std::make_shared<std::vector<Zone>>();

}

WildPatchWorkpp::~WildPatchWorkpp() {}

bool point_z_cmp(PointXYZ a, PointXYZ b) { return a.z < b.z; }

Eigen::MatrixX3d WildPatchWorkpp::toEigenCloud(const vector<PointXYZ> &cloud) {

    Eigen::MatrixX3d dst(cloud.size(), 3);
    int j=0;
    for (auto &p: cloud) {
        dst.row(j++) << p.x, p.y, p.z;
    }
    return dst;
}

Eigen::VectorXi WildPatchWorkpp::toIndices(const vector<PointXYZ> &cloud) {

    Eigen::VectorXi dst(cloud.size());
    int j = 0;
    for (auto &p: cloud) {
        dst.row(j++) << p.idx;
    }
    return dst;
}


int WildPatchWorkpp::Init(YAML::Node &config) {

    use_multi_lidar_ = config["patchworkpp_wild"]["use_multi_lidar"].as<bool>() ; 
    use_filter_ = config["patchworkpp_wild"]["use_filter"].as<bool>() ; 
    use_loc_trans_ = config["patchworkpp_wild"]["use_loc_trans"].as<bool>() ;
    use_loc_trans_remove_ =  config["patchworkpp_wild"]["use_loc_trans_remove"].as<bool>() ;

    use_lidar_detect_ = config["patchworkpp_wild"]["use_lidar_detect"].as<bool>() ;
    ncr_thresh_ = config["patchworkpp_wild"]["ncr_thresh"].as<double>() ; 
    detect_num_ = config["patchworkpp_wild"]["detect_num"].as<double>() ; 

    convert_count_thresh_ = config["patchworkpp_wild"]["convert_count_thresh"].as<int>() ;
    filter_nb_num_ = config["patchworkpp_wild"]["filter_nb_num"].as<double>() ; 
    filter_side_range_ = config["patchworkpp_wild"]["filter_side_range"].as<double>() ; 
    filter_front_range_ = config["patchworkpp_wild"]["filter_front_range"].as<double>() ; 
    filter_back_range_ = config["patchworkpp_wild"]["filter_back_range"].as<double>() ; 
    filter_nb_radius_ = config["patchworkpp_wild"]["filter_nb_radius"].as<double>() ; 
    convert_front_range_ = config["patchworkpp_wild"]["convert_front_range"].as<double>() ;
    convert_blind_side_range_ = config["patchworkpp_wild"]["convert_blind_side_range"].as<double>() ;
    convert_blind_front_range_ = config["patchworkpp_wild"]["convert_blind_front_range"].as<double>() ;
    std::cout << " convert_front_range_ " << convert_front_range_
              << " convert_blind_side_range_ " << convert_blind_side_range_
              << " convert_blind_front_range_ " << convert_blind_front_range_ 
              << std::endl;

    std::string loc_subscribe_topic = config["patchworkpp_wild"]["loc_subscribe"].as<std::string>() ;
    std::string lidar_subscribe_topic = config["patchworkpp_wild"]["lidar_subscribe"].as<std::string>() ;
    std::string lidar_subscribe_left_front_topic = config["patchworkpp_wild"]["lidar_subscribe_left_front"].as<std::string>() ;
    std::string lidar_subscribe_right_front_topic = config["patchworkpp_wild"]["lidar_subscribe_right_front"].as<std::string>() ;

    std::string lidar_subscribe_front_topic = config["patchworkpp_wild"]["lidar_subscribe_front"].as<std::string>() ;

    std::string lidar_subscribe_back_topic = config["patchworkpp_wild"]["lidar_subscribe_back"].as<std::string>() ;
    std::string odom_subscribe_topic = config["patchworkpp_wild"]["odom_subscribe"].as<std::string>() ;

    std::string pub_ground_topic = config["patchworkpp_wild"]["pub_ground"].as<std::string>() ;
    std::string pub_ground_revert_topic = config["patchworkpp_wild"]["pub_ground_revert"].as<std::string>() ;
    std::string pub_ground_intensity_topic = config["patchworkpp_wild"]["pub_ground_intensity"].as<std::string>() ;
    std::string pub_patch_topic = config["patchworkpp_wild"]["pub_patch"].as<std::string>() ;
    std::string pub_marker_topic = config["patchworkpp_wild"]["pub_marker"].as<std::string>() ;
    std::string pub_marker_init_topic = config["patchworkpp_wild"]["pub_marker_init"].as<std::string>() ;
    std::string pub_patch_init_topic = config["patchworkpp_wild"]["pub_patch_init"].as<std::string>() ;
    std::string pub_lidar_seg_ret_topic = config["patchworkpp_wild"]["pub_lidar_seg_ret"].as<std::string>() ;

    
    filter_ground_noise_ =  config["patchworkpp_wild"]["filter_ground_noise"].as<double>() ;
    min_x_range_ =  config["patchworkpp_wild"]["min_x_range_"].as<double>() ;
    max_x_range_ =  config["patchworkpp_wild"]["max_x_range_"].as<double>() ;
    min_y_range_ =  config["patchworkpp_wild"]["min_y_range_"].as<double>() ;
    max_y_range_ =  config["patchworkpp_wild"]["max_y_range_"].as<double>() ;
    min_z_range_ =  config["patchworkpp_wild"]["min_z_range_"].as<double>() ;  
    max_z_range_ =  config["patchworkpp_wild"]["max_z_range_"].as<double>() ;  
    car_front =  config["patchworkpp_wild"]["car_front"].as<double>() ;  
    car_back = config["patchworkpp_wild"]["car_back"].as<double>() ;  
    car_left = config["patchworkpp_wild"]["car_left"].as<double>() ;  
    car_right = config["patchworkpp_wild"]["car_right"].as<double>() ; 
    top_lidar_x_offset = config["patchworkpp_wild"]["top_lidar_x_offset"].as<double>() ;
    top_lidar_y_offset = config["patchworkpp_wild"]["top_lidar_y_offset"].as<double>() ;
    top_lidar_dis_thresh = config["patchworkpp_wild"]["top_lidar_dis_thresh"].as<double>() ;
    top_lidar_yaw_offset = config["patchworkpp_wild"]["top_lidar_yaw_offset"].as<double>() ;
    left_lidar_yaw_offset = config["patchworkpp_wild"]["left_lidar_yaw_offset"].as<double>() ;
    right_lidar_yaw_offset = config["patchworkpp_wild"]["right_lidar_yaw_offset"].as<double>() ;
    back_lidar_x_offset = config["patchworkpp_wild"]["back_lidar_x_offset"].as<double>() ;
    back_lidar_y_offset = config["patchworkpp_wild"]["back_lidar_y_offset"].as<double>() ;
    back_lidar_yaw_offset = config["patchworkpp_wild"]["back_lidar_yaw_offset"].as<double>() ;

    // 6DoF 标定变换参数
    top_x_offset_ = config["patchworkpp_wild"]["top_x_offset"].as<double>();
    top_y_offset_ = config["patchworkpp_wild"]["top_y_offset"].as<double>();
    top_z_offset_ = config["patchworkpp_wild"]["top_z_offset"].as<double>();
    top_yaw_      = config["patchworkpp_wild"]["top_yaw"].as<double>();
    top_pitch_    = config["patchworkpp_wild"]["top_pitch"].as<double>();
    top_roll_     = config["patchworkpp_wild"]["top_roll"].as<double>();

    front_x_offset_ = config["patchworkpp_wild"]["front_x_offset"].as<double>();
    front_y_offset_ = config["patchworkpp_wild"]["front_y_offset"].as<double>();
    front_z_offset_ = config["patchworkpp_wild"]["front_z_offset"].as<double>();
    front_yaw_      = config["patchworkpp_wild"]["front_yaw"].as<double>();
    front_pitch_    = config["patchworkpp_wild"]["front_pitch"].as<double>();
    front_roll_     = config["patchworkpp_wild"]["front_roll"].as<double>();

    back_x_offset_ = config["patchworkpp_wild"]["back_x_offset"].as<double>();
    back_y_offset_ = config["patchworkpp_wild"]["back_y_offset"].as<double>();
    back_z_offset_ = config["patchworkpp_wild"]["back_z_offset"].as<double>();
    back_yaw_      = config["patchworkpp_wild"]["back_yaw"].as<double>();
    back_pitch_    = config["patchworkpp_wild"]["back_pitch"].as<double>();
    back_roll_     = config["patchworkpp_wild"]["back_roll"].as<double>();

    std::cout << "[TopLidar]   calib offset(x,y,z)=(" << top_x_offset_ << "," << top_y_offset_ << "," << top_z_offset_
              << ") rpy(yaw,pitch,roll)=(" << top_yaw_ << "," << top_pitch_ << "," << top_roll_ << ")" << std::endl;
    std::cout << "[FrontLidar] calib offset(x,y,z)=(" << front_x_offset_ << "," << front_y_offset_ << "," << front_z_offset_
              << ") rpy(yaw,pitch,roll)=(" << front_yaw_ << "," << front_pitch_ << "," << front_roll_ << ")" << std::endl;
    std::cout << "[BackLidar]  calib offset(x,y,z)=(" << back_x_offset_ << "," << back_y_offset_ << "," << back_z_offset_
              << ") rpy(yaw,pitch,roll)=(" << back_yaw_ << "," << back_pitch_ << "," << back_roll_ << ")" << std::endl;

    std::cout << " P++ params_ min_x_range_  " <<  min_x_range_ << std::endl; 
    std::cout << " P++ params_ max_x_range_  " <<  max_x_range_ << std::endl; 
    std::cout << " P++ params_ min_y_range_  " <<  min_y_range_ << std::endl; 
    std::cout << " P++ params_ max_y_range_  " <<  max_y_range_ << std::endl; 
    std::cout << " P++ params_  use_filter_ " << use_filter_ << std::endl;
    // CNOTICE_LOG(" P++ params_ min_x_range_: %d", min_x_range_);
    // CNOTICE_LOG(" P++ params_ max_x_range_: %d", max_x_range_);
    // CNOTICE_LOG(" P++ params_ min_y_range_: %d", min_y_range_);
    // CNOTICE_LOG(" P++ params_ max_y_range_: %d", max_y_range_);

    // CNOTICE_LOG(" P++ params_ top_lidar_x_offset: %d", top_lidar_x_offset);
    // CNOTICE_LOG(" P++ params_ top_lidar_y_offset: %d", top_lidar_y_offset);
    // CNOTICE_LOG(" P++ params_ top_lidar_yaw_offset: %d", top_lidar_yaw_offset);
    // CNOTICE_LOG(" P++ params_ back_lidar_x_offset: %d", back_lidar_x_offset);
    // CNOTICE_LOG(" P++ params_ back_lidar_y_offset: %d", back_lidar_y_offset);
    // CNOTICE_LOG(" P++ params_ back_lidar_yaw_offset: %d", back_lidar_yaw_offset);

    auto config_ = seres::perception::LidarSegConfigType();
    config_.verbose = config["patchworkpp_wild"]["verbose"].as<bool>() ; 
    config_.enable_RNR = config["patchworkpp_wild"]["enable_RNR"].as<bool>() ; 
    config_.enable_RVPF = config["patchworkpp_wild"]["enable_RVPF"].as<bool>() ; 
    config_.enable_TGR = config["patchworkpp_wild"]["enable_TGR"].as<bool>() ; 
    config_.enable_MRT = config["patchworkpp_wild"]["enable_MRT"].as<bool>() ; 

    config_.num_iter = config["patchworkpp_wild"]["num_iter"].as<int>() ; 
    config_.num_lpr = config["patchworkpp_wild"]["num_lpr"].as<int>() ; 
    config_.num_min_pts = config["patchworkpp_wild"]["num_min_pts"].as<int>() ; 
    config_.num_zones = config["patchworkpp_wild"]["num_zones"].as<int>() ; 
    config_.num_rings_of_interest = config["patchworkpp_wild"]["num_rings_of_interest"].as<int>() ; 

    config_.RNR_ver_angle_thr = config["patchworkpp_wild"]["RNR_ver_angle_thr"].as<double>() ; 
    config_.RNR_intensity_thr = config["patchworkpp_wild"]["RNR_intensity_thr"].as<double>() ; 
    config_.sensor_height = config["patchworkpp_wild"]["sensor_height"].as<double>() ; 
    config_.th_seeds = config["patchworkpp_wild"]["th_seeds"].as<double>() ; 
    config_.th_dist = config["patchworkpp_wild"]["th_dist"].as<double>() ; 
    config_.obstacle_height = config["patchworkpp_wild"]["obstacle_height"].as<double>() ;
    config_.step_height = config["patchworkpp_wild"]["step_height"].as<double>() ; 
    config_.th_seeds_v = config["patchworkpp_wild"]["th_seeds_v"].as<double>() ; 
    config_.th_dist_v = config["patchworkpp_wild"]["th_dist_v"].as<double>() ; 
    config_.max_range = config["patchworkpp_wild"]["max_range"].as<double>() ; 
    config_.min_range = config["patchworkpp_wild"]["min_range"].as<double>() ; 
    config_.uprightness_thr = config["patchworkpp_wild"]["uprightness_thr"].as<double>() ; 
    config_.adaptive_seed_selection_margin = config["patchworkpp_wild"]["adaptive_seed_selection_margin"].as<double>() ; 
    config_.intensity_thr = config["patchworkpp_wild"]["intensity_thr"].as<double>() ; 


    config_.near_cal_range = config["patchworkpp_wild"]["near_cal_range"].as<int>() ; 
    config_.far_cal_range = config["patchworkpp_wild"]["far_cal_range"].as<int>() ; 

    config_.near_revert_thresh = config["patchworkpp_wild"]["near_revert_thresh"].as<double>() ; 
    config_.middle_revert_thresh = config["patchworkpp_wild"]["middle_revert_thresh"].as<double>() ; 
    config_.far_revert_thresh = config["patchworkpp_wild"]["far_revert_thresh"].as<double>() ; 
    config_.num_sectors_each_zone = config["patchworkpp_wild"]["num_sectors_each_zone"].as<std::vector<int>>() ;
    config_.num_rings_each_zone = config["patchworkpp_wild"]["num_rings_each_zone"].as<std::vector<int>>() ;

    config_.max_flatness_storage = config["patchworkpp_wild"]["max_flatness_storage"].as<int>() ; 
    config_.max_elevation_storage = config["patchworkpp_wild"]["max_elevation_storage"].as<int>() ; 

    config_.elevation_thr = config["patchworkpp_wild"]["elevation_thr"].as<std::vector<double>>() ;
    config_.flatness_thr = config["patchworkpp_wild"]["flatness_thr"].as<std::vector<double>>() ;

    params_ = config_;
    for (int i = 0; i < 8; ++i) {
        elevation_mean_[i] = -params_.sensor_height;
    }
    double min_range_z2_ = (7 * params_.min_range + params_.max_range) / 8.0;
    double min_range_z3_ = (3 * params_.min_range + params_.max_range) / 4.0;
    double min_range_z4_ = (params_.min_range + params_.max_range) / 2.0;
    min_ranges_ = {params_.min_range, min_range_z2_, min_range_z3_, min_range_z4_};

    ring_sizes_ = {(min_range_z2_ - params_.min_range) / params_.num_rings_each_zone.at(0),
                    (min_range_z3_ - min_range_z2_) / params_.num_rings_each_zone.at(1),
                    (min_range_z4_ - min_range_z3_) / params_.num_rings_each_zone.at(2),
                    (params_.max_range - min_range_z4_) / params_.num_rings_each_zone.at(3)};
    sector_sizes_ = {2 * M_PI / params_.num_sectors_each_zone.at(0),
                        2 * M_PI / params_.num_sectors_each_zone.at(1),
                        2 * M_PI / params_.num_sectors_each_zone.at(2),
                        2 * M_PI / params_.num_sectors_each_zone.at(3)};

    
    for (int k = 0; k < params_.num_zones; k++) {
        
        Ring empty_ring;
        empty_ring.resize(params_.num_sectors_each_zone[k]);

        Zone z;
        for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
            z.push_back(empty_ring);
        }

        ConcentricZoneModel_.push_back(z);
        // ConcentricZoneModel_thread->push_back(z);
        ConcentricZoneModel_ground_.push_back(z);
        ConcentricZoneModel_nonground_.push_back(z);
        
        // neighbour init
        Ring_neighbour empty_ring_;
        empty_ring_.resize(params_.num_sectors_each_zone[k]);
        
        Zone_neighbour z_;
        for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
            z_.push_back(empty_ring_);
        }
        ConcentricZoneModel_neighbour_.push_back(z_);

        // center init
        Ring_center empty_ring_center;
        empty_ring_center.resize(params_.num_sectors_each_zone[k]);
        
        Zone_center z_center;
        for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
            z_center.push_back(empty_ring_center);
        }
        ConcentricZoneModel_center_.push_back(z_center);

    }

    // 绘制自车大小
    visualization_msgs::msg::Marker self_marker;
    self_marker.header.frame_id = vis_frame_id;
    self_marker.header.stamp = node_->get_clock()->now() ;
    self_marker.ns = "p++_wild_init_";
    self_marker.id = 0;
    self_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;  // 绘制曲线
    self_marker.action = visualization_msgs::msg::Marker::ADD;
    self_marker.scale.x = 0.1;  // 曲线粗细
    self_marker.color.r = 1.0f;
    self_marker.color.g = 0.0f;
    self_marker.color.b = 0.0f;
    self_marker.color.a = 1.0f;
    self_marker.lifetime = rclcpp::Duration::from_seconds(0);

    geometry_msgs::msg::Point point;
    point.x = car_front;
    point.y = car_left;
    point.z = 1.0;
    self_marker.points.push_back(point);

    point.x = car_front;  
    point.y = car_right;
    point.z = 1.0;
    self_marker.points.push_back(point);

    point.x = car_back;  
    point.y = car_right;
    point.z = 1.0;
    self_marker.points.push_back(point);


    point.x = car_back;  
    point.y = car_left;
    point.z = 1.0;
    self_marker.points.push_back(point);

    
    point.x = car_front;
    point.y = car_left;
    point.z = 1.0;
    self_marker.points.push_back(point);

    mark_array_poly.markers.push_back(self_marker);
    // caculate neighbour idx of each sector
    // 计算每个patch的中心点
    pcl::PointCloud<pcl::PointXYZRGB> patch_viz_cloud;

    double max_range = params_.max_range, min_range = params_.min_range;
    double min_range_0 = min_ranges_[0], min_range_1 = min_ranges_[1], min_range_2 = min_ranges_[2], min_range_3 = min_ranges_[3];
    int num_ring_0 = params_.num_rings_each_zone[0], num_sector_0 = params_.num_sectors_each_zone[0];
    int num_ring_1 = params_.num_rings_each_zone[1], num_sector_1 = params_.num_sectors_each_zone[1];
    int num_ring_2 = params_.num_rings_each_zone[2], num_sector_2 = params_.num_sectors_each_zone[2];
    int num_ring_3 = params_.num_rings_each_zone[3], num_sector_3 = params_.num_sectors_each_zone[3];
    int abs_ring_idx = -1;
    int patch_count = 0;
    for (int zone_idx = 0; zone_idx < params_.num_zones; ++zone_idx) {
        // 一共四个大圆环区域，取其中一个圆环
        // auto zone_ground = ConcentricZoneModel_neighbour_[zone_idx];
        // 每个大圆环区域有num_rings_each_zone个小圆环区域ring
        for (int ring_idx = 0; ring_idx < params_.num_rings_each_zone[zone_idx]; ++ring_idx) {
            abs_ring_idx += 1;
            // 每个小圆环区域ring有num_sectors_each_zone个patch，称为sector
            for (int sector_idx = 0; sector_idx < params_.num_sectors_each_zone[zone_idx]; ++sector_idx) {

                patch_count ++;

                double zone_min_range = min_ranges_[zone_idx];
                double r_len = ring_idx * ring_sizes_[zone_idx] + zone_min_range;
                double angle = sector_idx * sector_sizes_[zone_idx]; // 从0度开始所以不用加0

                pcl::PointXYZRGB point_1;
                //RL 下面那条线 右下点
                point_1.x = r_len * cos(angle);
                point_1.y = r_len * sin(angle);
                point_1.z = 0.0;
                point_1.r = 0;
                point_1.g = 255;
                point_1.b = 0;
                patch_viz_cloud.points.push_back(point_1);

                // RL+1 下面那条线 右下点的下一点
                pcl::PointXYZRGB point_3;
                double angle_next = angle + sector_sizes_[zone_idx];
                point_3.x = r_len * cos(angle_next);
                point_3.y = r_len * sin(angle_next);
                point_3.z = 0.0;
                point_3.r = 0;
                point_3.g = 255;
                point_3.b = 0;
                patch_viz_cloud.points.push_back(point_3);

                // RU 上面那条线 右上点
                pcl::PointXYZRGB point_2;
                double r_len_next = r_len + ring_sizes_[zone_idx];
                point_2.x = r_len_next * cos(angle);
                point_2.y = r_len_next * sin(angle);
                point_2.z = 0.0;
                point_2.r = 0;
                point_2.g = 255;
                point_2.b = 0;
                patch_viz_cloud.points.push_back(point_2);



                // RU+1 上面那条线 右上点的下一点
                pcl::PointXYZRGB point_4;
                point_4.x = r_len_next * cos(angle_next);
                point_4.y = r_len_next * sin(angle_next);
                point_4.z = 0.0;
                point_4.r = 0;
                point_4.g = 255;
                point_4.b = 0;
                patch_viz_cloud.points.push_back(point_4);

                // patch中心点
                pcl::PointXYZRGB point_5;
                point_5.x = (point_1.x + point_2.x + point_3.x + point_4.x )/4;
                point_5.y = (point_1.y + point_2.y + point_3.y + point_4.y )/4;
                point_5.z = 0;
                point_5.r = 255;
                point_5.g = 255;
                point_5.b = 0;
                patch_viz_cloud.points.push_back(point_5);

                ConcentricZoneModel_center_[zone_idx][ring_idx][sector_idx].push_back(patch_center(point_5.x,point_5.y));


                // 正常来说一个sector有8个邻居左上，上，右上，左，右，下左，下，下右
                int left_sector,right_sector,up_ring_idx,down_ring_idx,up_zone_idx,down_zone_idx;
                up_zone_idx = floor((abs_ring_idx+1)/4);
                down_zone_idx = floor((abs_ring_idx-1)/4);
                up_ring_idx =((abs_ring_idx+1)%4);
                down_ring_idx = ((abs_ring_idx-1)%4);

                if(sector_idx > 0 && sector_idx < params_.num_sectors_each_zone[zone_idx]-1){
                    left_sector = sector_idx + 1;
                    right_sector = sector_idx - 1;
                }else if(sector_idx == params_.num_sectors_each_zone[zone_idx] - 1){
                    left_sector = 0;
                    right_sector = sector_idx - 1;
                }else if(abs(sector_idx) < 0.000001  ){
                    left_sector = sector_idx + 1;
                    right_sector =  params_.num_sectors_each_zone[zone_idx] - 1;
                }

                // TODO 上下边界判断滤除
                if(abs_ring_idx!=0 && abs_ring_idx!=7 && abs_ring_idx!=8 && abs_ring_idx!=11 && abs_ring_idx!=12 && abs_ring_idx!=15){

                    if(up_zone_idx < params_.num_zones && down_zone_idx > -1 && down_ring_idx > -1 && up_ring_idx < 4 ){
                        // 左上
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,left_sector));
                        // 上
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,sector_idx));
                        // 右上
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,right_sector));
                        // 左
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,left_sector));
                        // 右
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,right_sector));
                        // 下左
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,left_sector));
                        // 下
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,sector_idx));
                        // 下右
                        ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,right_sector));
                    }


                }else if(abs_ring_idx==7 || abs_ring_idx==11 || abs_ring_idx==15 ){ // 不要上
                    // // 左上
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,left_sector*2-1));
                    // // 上
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,sector_idx*2));
                    // // 右上
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,right_sector));
                    // 左
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,left_sector));
                    // 右
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,right_sector));
                    // 下左
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,left_sector));
                    // 下
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,sector_idx));
                    // 下右
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,right_sector));
                }else if(abs_ring_idx==8 || abs_ring_idx==0 || abs_ring_idx==12){// 不要下
                    // 左上
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,left_sector));
                    // 上
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,sector_idx));
                    // 右上
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,right_sector));
                    // 左
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,left_sector));
                    // 右
                    ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,right_sector));
                    // // 下左
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,left_sector));
                    // // 下
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,sector_idx));
                    // // 下右
                    // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,right_sector));
                }
                // else if(abs_ring_idx==11){// 不要上
                //     // // 左上
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,left_sector*2-1));
                //     // // 上
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,sector_idx*2));
                //     // // 右上
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,right_sector));
                //     // 左
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,left_sector));
                //     // 右
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,right_sector));
                //     // 下左
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,left_sector));
                //     // 下
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,sector_idx));
                //     // 下右
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,right_sector));
                // }else if(abs_ring_idx==12){// 不要下
                //     // // 左上
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,left_sector));
                //     // 上
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,sector_idx));
                //     // 右上
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(up_zone_idx,up_ring_idx,right_sector));
                //     // 左
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,left_sector));
                //     // 右
                //     ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(zone_idx,ring_idx,right_sector));
                //     // 下左
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,left_sector));
                //     // // 下
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,sector_idx));
                //     // // 下右
                //     // ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx].push_back(neighbour_idx(down_zone_idx,down_ring_idx,right_sector));
                // }


                visualization_msgs::msg::Marker txt_marker;
                txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                txt_marker.action = visualization_msgs::msg::Marker::ADD;
                txt_marker.header.frame_id = vis_frame_id;
                txt_marker.ns = "p++_wild_init";
                txt_marker.header.stamp =  node_->get_clock()->now();
                // txt_marker.scale.x = 0.5;
                // txt_marker.scale.y = 0.5;
                txt_marker.scale.z = 0.2;
                txt_marker.color.r = 1.0;
                txt_marker.color.g = 1.0;
                txt_marker.color.b = 1.0;
                txt_marker.color.a = 0.9;
                txt_marker.id = patch_count;
                txt_marker.pose.position.x = ConcentricZoneModel_center_[zone_idx][ring_idx][sector_idx][0].center_x;
                txt_marker.pose.position.y = ConcentricZoneModel_center_[zone_idx][ring_idx][sector_idx][0].center_y;
                txt_marker.pose.position.z = 0;
                txt_marker.lifetime = rclcpp::Duration::from_seconds(0);
                std::stringstream str;

                str << std::setprecision(1) << "c:" <<  zone_idx << "/" <<ring_idx << "/" << sector_idx << "\n";
                for(auto nb:ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx]){
                    str << "nb:" << nb.zone_idx << "/" << nb.ring_idx << "/" << nb.sector_idx << "\n";
                }
                // str << std::setprecision(1) << "c:" <<  zone_idx << "/" <<ring_idx << "/" << sector_idx
                //                         // << "\nul:" << up_zone_idx << "/" << up_ring_idx << "/" << left_sector << "|"
                //                         // << "u:" << up_zone_idx << "/" << up_ring_idx << "/" << sector_idx << "|"
                //                         // << "ur:" << up_zone_idx << "/" << up_ring_idx << "/" << right_sector << "|"
                //                         // << "\ndl:" << down_zone_idx << "/" << down_ring_idx << "/" << left_sector << "|"
                //                         // << "d:" << down_zone_idx << "/" << down_ring_idx << "/" << sector_idx << "|"
                //                         // << "dr:" << down_zone_idx << "/" << down_ring_idx << "/" << right_sector << "|"

                //                         << "\nl/r:"<< left_sector << "/" <<  right_sector
                //                            <<"/" << zone_idx * params_.num_rings_each_zone[zone_idx] + ring_idx 
                //                         << "\nu::" << up_zone_idx  << "/" << up_ring_idx
                //                         << "\nd::" << down_zone_idx << "/" << down_ring_idx
                //                         ;

                visualization_msgs::msg::Marker curve_marker;
                curve_marker.header.frame_id = vis_frame_id;
                curve_marker.header.stamp = node_->get_clock()->now() ;
                curve_marker.ns = "p++_wild_init_";
                curve_marker.id = patch_count;
                curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;  // 绘制曲线
                curve_marker.action = visualization_msgs::msg::Marker::ADD;
                curve_marker.scale.x = 0.02;  // 曲线粗细
                curve_marker.color.r = 1.0f;
                curve_marker.color.g = 1.0f;
                curve_marker.color.b = 1.0f;
                curve_marker.color.a = 1.0f;
                curve_marker.lifetime = rclcpp::Duration::from_seconds(0);

                geometry_msgs::msg::Point point;
                point.x = point_1.x;  // 生成一个圆形曲线
                point.y = point_1.y;
                point.z = point_1.z;
                curve_marker.points.push_back(point);

                point.x = point_2.x;  // 生成一个圆形曲线
                point.y = point_2.y;
                point.z = point_2.z;
                curve_marker.points.push_back(point);

                point.x = point_4.x;  // 生成一个圆形曲线
                point.y = point_4.y;
                point.z = point_4.z;
                curve_marker.points.push_back(point);

                point.x = point_3.x;  // 生成一个圆形曲线
                point.y = point_3.y;
                point.z = point_3.z;
                curve_marker.points.push_back(point);
                // // 曲线的多个点
                // for (double t = 0; t < 2 * 3.14159; t += 0.1)
                // {
                //     geometry_msgs::msg::Point point;
                //     point.x = 2.0 * cos(t);  // 生成一个圆形曲线
                //     point.y = 2.0 * sin(t);
                //     point.z = 0.0;
                //     curve_marker.points.push_back(point);
                // }

                txt_marker.text = str.str();
                mark_array_init_txt.markers.push_back(txt_marker);
                mark_array_poly.markers.push_back(curve_marker);
            }
        }
    }
    sensor_msgs::msg::PointCloud2 cloud_ROS;
    pcl::toROSMsg(patch_viz_cloud, cloud_ROS);
    cloud_ROS.height = 1;
    cloud_ROS.width = patch_viz_cloud.points.size();
    cloud_ROS.header.stamp = node_->get_clock()->now();
    cloud_ROS.header.frame_id = vis_frame_id;
    


    // update_flatness_.reserve(params_.num_rings_of_interest,0.0);
    // update_elevation_.reserve(params_.num_rings_of_interest,0.0);

    // std::cout << "PatchWorkpp::PatchWorkpp() - INITIALIZATION COMPLETE" << std::endl;
    // std::cout << " P++ ConcentricZoneModel_ size " << ConcentricZoneModel_.size() <<std::endl;
    // std::cout << " P++ params_ TGR " << params_.enable_TGR <<std::endl;
    // std::cout << " P++ params_ min_range_z2_  " <<  min_range_z2_ << std::endl;
    // std::cout << " P++ params_ min_range_z3_  " <<  min_range_z3_ << std::endl;
    // std::cout << " P++ params_ min_range_z4_  " <<  min_range_z4_ << std::endl;
    // std::cout << "ring_sizes_: ";
    // for (auto ch : ring_sizes_) {
    //     std::cout << ch << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "sector_sizes_: ";
    // for (auto ch : sector_sizes_) {
    //     std::cout << ch << " ";
    // }
    // std::cout << std::endl;

    // std::cout << " P++ params_ use_multi_lidar_   " << use_multi_lidar_ << std::endl;

    // CNOTICE_LOG ("PatchWorkpp::PatchWorkpp() - INITIALIZATION COMPLETE") ;
    // CNOTICE_LOG (" P++ ConcentricZoneModel_ size : %d", ConcentricZoneModel_.size());
    // CNOTICE_LOG (" P++ params_ TGR : %d", params_.enable_TGR );
    // CNOTICE_LOG (" P++ params_ min_range_z2_ : %d ",min_range_z2_ );
    // CNOTICE_LOG ( " P++ params_ min_range_z3_ : %d  " , min_range_z3_);
    // CNOTICE_LOG (" P++ params_ min_range_z4_ : %d " , min_range_z4_);
    std::string show_size = " ";
    for (auto ch : ring_sizes_) {
        show_size += std::to_string(ch) ;
        show_size += " " ;
    }
    // CNOTICE_LOG (" ring_sizes_ : %s " , show_size.c_str());

    std::string show_sector_size = " ";
    for (auto ch : sector_sizes_) {
       show_sector_size += std::to_string(ch);
       show_sector_size == " ";
    }
    // CNOTICE_LOG (" sector_size_ : %s " , show_sector_size.c_str());

    // CNOTICE_LOG (" P++ params_ use_multi_lidar_ : %d ", use_multi_lidar_ );
    // CNOTICE_LOG (" P++ params_ use_filter_ : %d ", use_filter_ );
    // CNOTICE_LOG (" P++ params_ filter_nb_num_ : %f ", filter_nb_num_ );
    // CNOTICE_LOG (" P++ params_ filter_nb_radius_ : %f ", filter_nb_radius_ );

    lidar_subscribe_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_subscribe_topic, 10, std::bind(&WildPatchWorkpp::LidarCallback, this, std::placeholders::_1));
    lidar_subscribe_front_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_subscribe_front_topic, 10, std::bind(&WildPatchWorkpp::LidarFrontCallback, this, std::placeholders::_1));
    lidar_subscribe_back_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_subscribe_back_topic, 10, std::bind(&WildPatchWorkpp::LidarBackCallback, this, std::placeholders::_1));
    odom_subscribe_ = node_->create_subscription<nav_msgs::msg::Odometry>(odom_subscribe_topic, 10, std::bind(&WildPatchWorkpp::OdomCallback, this, std::placeholders::_1));
    std::cout << "Subscribed odom: " << odom_subscribe_topic << std::endl;
    // lidar_subscribe_ = node_->create_subscription<serespilot_msgs::msg::PointCloud2>(lidar_subscribe_topic, 10, std::bind(&WildPatchWorkpp::LidarCallback, this, std::placeholders::_1));

    // loc_subscribe_ = node_->create_subscription<serespilot_msgs::msg::Hcinspvatzcb>(loc_subscribe_topic, 10, std::bind(&WildPatchWorkpp::HcLocalizationCallback, this, std::placeholders::_1));

    
    
    // lidar_subscribe_left_ = node_->create_subscription<serespilot_msgs::msg::PointCloud2>(lidar_subscribe_left_front_topic, 10, std::bind(&WildPatchWorkpp::LidarLeftCallback, this, std::placeholders::_1));

    // lidar_subscribe_right_ = node_->create_subscription<serespilot_msgs::msg::PointCloud2>(lidar_subscribe_right_front_topic, 10, std::bind(&WildPatchWorkpp::LidarRightCallback, this, std::placeholders::_1));

    // lidar_subscribe_front_ = node_->create_subscription<serespilot_msgs::msg::PointCloud2>(lidar_subscribe_front_topic, 10, std::bind(&WildPatchWorkpp::LidarFrontCallback, this, std::placeholders::_1));

    // lidar_subscribe_back_ = node_->create_subscription<serespilot_msgs::msg::PointCloud2>(lidar_subscribe_back_topic, 10, std::bind(&WildPatchWorkpp::LidarBackCallback, this, std::placeholders::_1));

    pub_ground    = node_->create_publisher<sensor_msgs::msg::PointCloud2>(pub_ground_topic, 10);
    pub_ground_revert    = node_->create_publisher<sensor_msgs::msg::PointCloud2>(pub_ground_revert_topic, 10);

    pub_ground_convert    = node_->create_publisher<sensor_msgs::msg::PointCloud2>("patchworkpp/ground_convert", 10);

    pub_ncr = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ncr", 10);

    pub_ground_intensity    = node_->create_publisher<sensor_msgs::msg::PointCloud2>(pub_ground_intensity_topic, 10);
    pub_patch    = node_->create_publisher<sensor_msgs::msg::PointCloud2>(pub_patch_topic, 10);
    pub_marker_=  node_->create_publisher<visualization_msgs::msg::MarkerArray>(pub_marker_topic, 10);

    pub_marker_init =  node_->create_publisher<visualization_msgs::msg::MarkerArray>(pub_marker_init_topic, 10);

    pub_patch_init = node_->create_publisher<visualization_msgs::msg::MarkerArray>(pub_patch_init_topic, 10);

    // pub_patch_init->publish(mark_array_poly);
    // pub_marker_init->publish(mark_array_init_txt);

    // pub_lidar_seg_ret    = node_->create_publisher<serespilot_msgs::msg::LidarSeg>(pub_lidar_seg_ret_topic, 10);


    return 1;
}


void WildPatchWorkpp::pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm) {

    double max_range = params_.max_range, min_range = params_.min_range;
    double min_range_0 = min_ranges_[0], min_range_1 = min_ranges_[1], min_range_2 = min_ranges_[2], min_range_3 = min_ranges_[3];
    int num_ring_0 = params_.num_rings_each_zone[0], num_sector_0 = params_.num_sectors_each_zone[0];
    int num_ring_1 = params_.num_rings_each_zone[1], num_sector_1 = params_.num_sectors_each_zone[1];
    int num_ring_2 = params_.num_rings_each_zone[2], num_sector_2 = params_.num_sectors_each_zone[2];
    int num_ring_3 = params_.num_rings_each_zone[3], num_sector_3 = params_.num_sectors_each_zone[3];

    for (int i=0; i<src.points.size(); i++) {

        double x = src.points[i].x, y = src.points[i].y, z = src.points[i].z;
        int intensity = src.points[i].intensity ;
        // std::cout << " src.row(i)(3) " << intensity << std::endl;
        // if(src.row(i)(3) >= 0 && src.row(i)(3) <= 255){
        //     intensity = src.row(i)(3);
        // }
        

        if ( z == std::numeric_limits<double>::min() ) continue;

        double r = xy2radius(x, y);
        int ring_idx, sector_idx;
        if ((r <= max_range) && (r > min_range)) {
            // double theta = xy2theta(pt.x, pt.y);
            double theta = xy2theta(x, y);
            
            if (r < min_range_1) { // In First rings
                ring_idx = min(static_cast<int>(((r - min_range_0) / ring_sizes_[0])), num_ring_0 - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[0])), num_sector_0 - 1);
                czm[0][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i,intensity));
            } else if (r < min_range_2) {
                ring_idx = min(static_cast<int>(((r - min_range_1) / ring_sizes_[1])), num_ring_1 - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[1])), num_sector_1 - 1);
                czm[1][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i,intensity));
            } else if (r < min_range_3) {
                ring_idx = min(static_cast<int>(((r - min_range_2) / ring_sizes_[2])), num_ring_2 - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[2])), num_sector_2 - 1);
                czm[2][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i,intensity));
            } else { // Far!
                ring_idx = min(static_cast<int>(((r - min_range_3) / ring_sizes_[3])), num_ring_3 - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[3])), num_sector_3 - 1);
                czm[3][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i,intensity));
            }

        } else {
            cloud_nonground_.push_back(PointXYZ(x, y, z, i,intensity));
        }
    }
    // if (params_.verbose) cout << "\033[1;33m" << "PatchWorkpp::pc2czm() - Divides pointcloud into the concentric zone model successfully" << "\033[0m" << endl;
    // CNOTICE_LOG( "PatchWorkpp::pc2czm() - Divides pointcloud into the concentric zone model successfully");
}

// For adaptive
void WildPatchWorkpp::extract_piecewiseground(
        const int zone_idx, const vector<PointXYZ> &src,
        vector<PointXYZ> &dst,
        vector<PointXYZ> &non_ground_dst) {
    
    // 0. Initialization
    if (!ground_pc_.empty()) ground_pc_.clear();
    if (!dst.empty()) dst.clear();
    if (!non_ground_dst.empty()) non_ground_dst.clear();

    // 1. Region-wise Vertical Plane Fitting (R-VPF) 
    // : removes potential vertical plane under the ground plane
    vector<PointXYZ> src_wo_verticals;
    src_wo_verticals = src;
    
    // if (params_.enable_RVPF)
    // {
    //     for (int i = 0; i < params_.num_iter; i++)
    //     {
    //         extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_, params_.th_seeds_v);
    //         estimate_plane(ground_pc_);

    //         if (zone_idx == 0 && normal_(2) < params_.uprightness_thr)
    //         {
    //             vector<PointXYZ> src_tmp;
    //             src_tmp = src_wo_verticals;
    //             src_wo_verticals.clear();

    //             for ( auto point : src_tmp ) {

    //                 double distance = calc_point_to_plane_d(point, normal_, d_);
                    
    //                 if ( abs(distance) < params_.th_dist_v ) {
    //                     non_ground_dst.push_back(point);
    //                 } else {
    //                     src_wo_verticals.push_back(point);
    //                 }
    //             }
    //         }
    //         else break;
    //     }
    // }

    // 2. Region-wise Ground Plane Fitting (R-GPF)
    // : fits the ground plane

    extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_);
    // 估计出一个初始平面的参数
    estimate_plane(ground_pc_);

    for (int i = 0; i < params_.num_iter; i++) {

        ground_pc_.clear();

        for ( auto point : src_wo_verticals ) {
            // 计算点到估计平面的距离
            double distance = calc_point_to_plane_d(point, normal_, d_);
            
            if (i < params_.num_iter - 1) {
                if ( distance < params_.th_dist ) {
                    ground_pc_.push_back(point);
                }
            } else {
                if ( distance < params_.th_dist ) {
                    // 最后一次估计后，小于阈值的点会被看成地面点
                    dst.push_back(point);
                } else {
                    // 最后一次估计后，大于阈值的点会被看成非地面点
                    non_ground_dst.push_back(point);
                }
            }
        }

        if (i < params_.num_iter -1) {
            //  i < n,估计过程中认为是地面的点
            estimate_plane(ground_pc_);
        }
        else {
            // i = n ，确定该区域的平面参数
            estimate_plane(dst);
        }
    }

    if ( dst.size() + non_ground_dst.size() != src.size() ) {
        cout << "\033[1;33m" << "Points are Missing/Adding !!! Please Check !! " << "\033[0m" << endl;
        cout << "gnd size: " << dst.size() << ", non gnd size: " << non_ground_dst.size() << ", src: " << src.size() << endl;
    }
}

void WildPatchWorkpp::temporal_ground_revert(std::vector<double> ring_flatness, std::vector<seres::perception::RevertCandidate> candidates,
                                         int concentric_idx)
{
    if (params_.verbose) {
        // CNOTICE_LOG("=========== Temporal Ground Revert (TGR) ===========");
    }
    
    // std::cout << "\033[1;34m" << "=========== Temporal Ground Revert (TGR) ===========" << "\033[0m" << endl;

    double mean_flatness = 0.0, stdev_flatness = 0.0;
    calc_mean_stdev(ring_flatness, mean_flatness, stdev_flatness);
    
    if (params_.verbose)
    {
        // cout << "[" << candidates[0].concentric_idx << ", " << candidates[0].sector_idx << "]"
        //      << " mean_flatness: " << mean_flatness << ", stdev_flatness: " << stdev_flatness << std::endl;
        // CNOTICE_LOG( "[ %d , %d mean_flatness : %f , stdev_flatness , %f ", candidates[0].concentric_idx , candidates[0].sector_idx , mean_flatness , stdev_flatness );

    }

    for( auto candidate : candidates )
    {        
        // Debug
        if(params_.verbose)
        {
            // cout << "\033[1;33m" << candidate.sector_idx << "th flat_sector_candidate"
            //      << " / flatness: " << candidate.ground_flatness
            //      << " / line_variable: " << candidate.line_variable
            //      << " / ground_num : " << candidate.regionwise_ground.size() 
            //      << "\033[0m" << endl;

            // CNOTICE_LOG(" %d , th flat_sector_candidate / flatness: %f / line_variable: %f  / ground_num %d ",candidate.sector_idx,candidate.ground_flatness,candidate.line_variable,candidate.regionwise_ground.size()  );
        }

        // 通过 mean_flatness 和 stdev_flatness 计算一个阈值 mu_flatness。然后通过 candidate.ground_flatness 计算候选区域的平整度的概率 prob_flatness。该计算基于 sigmoid 函数，平整度与期望值之间的差距越大，概率值越低。
        double mu_flatness = mean_flatness + 1.5*stdev_flatness;
        double prob_flatness = 1/(1+exp( (candidate.ground_flatness-mu_flatness)/(mu_flatness/10) ));

        // 如果候选区域内地面点数大于 1500 且平整度低于某个阈值（由 params_.th_dist 决定），则直接将该区域的平整度概率设置为 1.0，意味着该区域肯定是地面。
        if (candidate.regionwise_ground.size() > 1500 && candidate.ground_flatness < params_.th_dist*params_.th_dist) prob_flatness = 1.0;
       
        double prob_line = 1.0;
        if (candidate.line_variable > 8.0 )//&& candidate.line_dir > M_PI/4)//
        {
            // if (params_.verbose) cout << "line_dir: " << candidate.line_dir << endl;
            prob_line = 0.0;
        }

        bool revert = prob_line*prob_flatness > 0.5;

        if ( concentric_idx < params_.num_rings_of_interest )
        {
            
            int zone_idx = floor(concentric_idx/4);
            int ring_idx = concentric_idx%4;

            // std::cout << " TGR : " << " concentric_idx " << concentric_idx << " | "
            //                        << " zone_idx " << zone_idx << " | "
            //                        << " ring_idx " << ring_idx << " | "
            //                        << " prob_line " << prob_line << " | "
            //                        << " prob_flatness " << prob_flatness << " | "
            //                        << " revert " << revert 
            //                        << std::endl;


            if (revert)
            {
                if (params_.verbose)
                {
                    // cout << "\033[1;32m" << "REVERT TRUE" << "\033[0m" << endl;
                    // CNOTICE_LOG("REVERT TRUE");
                }
                addCloud(cloud_ground_, candidate.regionwise_ground);
                addCloud(cloud_revert, candidate.regionwise_ground);


                addCloud(ConcentricZoneModel_ground_[zone_idx][ring_idx][candidate.sector_idx],candidate.regionwise_ground);
                

            }
            else
            {
                if (params_.verbose) 
                {
                    cout << "\033[1;31m" << "FINAL REJECT" << "\033[0m" << endl;
                    // CNOTICE_LOG("FINAL REJECT");
                }
                addCloud(cloud_nonground_, candidate.regionwise_ground);
                addCloud(ConcentricZoneModel_nonground_[zone_idx][ring_idx][candidate.sector_idx],candidate.regionwise_ground);
            }                
        }
    }

}

void WildPatchWorkpp::update_flatness_thr(void)
{
    for (int i=0; i<params_.num_rings_of_interest; i++)
    {
        if (update_flatness_[i].empty()) break;
        if (update_flatness_[i].size() <= 1) break;

        double update_mean = 0.0, update_stdev = 0.0;
        calc_mean_stdev(update_flatness_[i], update_mean, update_stdev);
        params_.flatness_thr[i] = update_mean+update_stdev;

        // if (params_.verbose) { cout << "flatness threshold [" << i << "]: " << params_.flatness_thr[i] << endl; }

        int exceed_num = update_flatness_[i].size() - params_.max_flatness_storage;
        if (exceed_num > 0) update_flatness_[i].erase(update_flatness_[i].begin(), update_flatness_[i].begin() + exceed_num);
    }

    return;
}

void WildPatchWorkpp::update_elevation_thr(void)
{
   for (int i=0; i<params_.num_rings_of_interest; i++)
    {
        if (update_elevation_[i].empty()) continue;

        double update_mean = 0.0, update_stdev = 0.0;
        calc_mean_stdev(update_elevation_[i], update_mean, update_stdev);
        elevation_mean_[i] = update_mean;
        if (i==0) {
            params_.elevation_thr[i] = update_mean + 3*update_stdev;
            params_.sensor_height = -update_mean;
        }
        else params_.elevation_thr[i] = update_mean + 2*update_stdev;

        // if (params_.verbose) cout << "elevation threshold [" << i << "]: " << params_.elevation_thr[i] << endl;

        int exceed_num = update_elevation_[i].size() - params_.max_elevation_storage;
        if (exceed_num > 0) update_elevation_[i].erase(update_elevation_[i].begin(), update_elevation_[i].begin() + exceed_num);
    }

    return;
}


void WildPatchWorkpp::flush_patches(vector<Zone> &czm) {

    for (int k = 0; k < params_.num_zones; k++) {
        for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
            for (int j = 0; j < params_.num_sectors_each_zone[k]; j++) {
                // czm[k][i][j].resize(MAX_POINTS, 3);
                czm[k][i][j].clear();
            }
        }
    }

    // if( params_.verbose ) CNOTICE_LOG("PatchWorkpp::flush_patches() - Flushed patches successfully!");
    if (params_.verbose) {
        cout << "\033[1;31m" << "PatchWorkpp::flush_patches() - Flushed patches successfully!" << "\033[0m" << endl;
    }
}

void WildPatchWorkpp::estimate_plane(const vector<PointXYZ> &ground) {

    if (ground.empty()) return;

    Eigen::MatrixX3d eigen_ground(ground.size(), 3);
    int j=0;
    for (auto &p: ground) {
        eigen_ground.row(j++) << p.x, p.y, p.z;
    }
    Eigen::MatrixX3d centered = eigen_ground.rowwise() - eigen_ground.colwise().mean();
    Eigen::MatrixX3d cov = (centered.adjoint() * centered) / double(eigen_ground.rows() - 1);
    
    pc_mean_.resize(3);
    pc_mean_ << eigen_ground.colwise().mean()(0), eigen_ground.colwise().mean()(1), eigen_ground.colwise().mean()(2);

    Eigen::JacobiSVD<Eigen::MatrixX3d> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();
       
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // normal_ 是平面的法向量，取自奇异值分解结果中的最小奇异值对应的右奇异向量 svd.matrixU().col(2)。
    
    // 如果法向量 normal_ 的 z 分量小于0（即法向量指向平面下面），则将法向量的方向反向（乘以 -1），确保法向量指向上方。
    if (normal_(2) < 0) { for(int i=0; i<3; i++) normal_(i) *= -1; }

    // mean ground seeds value
    Eigen::Vector3d seeds_mean = pc_mean_.head<3>();

    // according to normal.T*[x,y,z] = -d
    //通过平面方程 normal_ . transpose() * [x, y, z] = -d 计算平面方程的常数项 d_。该值通过点云的均值（重心）与法向量的点积计算得出。
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);
}


// // R - VPF采用的初始化种子点的方法
// void WildPatchWorkpp::extract_initial_seeds(
//         const int zone_idx, const vector<PointXYZ> &p_sorted,
//         vector<PointXYZ> &init_seeds, double th_seed) {
    
//     init_seeds.clear();

//     // LPR is the mean of low point representative
//     double sum = 0;
//     int cnt = 0;

//     int init_idx = 0;
//     if (zone_idx == 0) {
//         for (int i = 0; i < p_sorted.size(); i++) {
//             if (p_sorted[i].z < params_.adaptive_seed_selection_margin * params_.sensor_height) {
//                 ++init_idx;
//             } else {
//                 break;
//             }
//         }
//     }
    
//     // Calculate the mean height value.
//     for (int i = init_idx; i < p_sorted.size() && cnt < params_.num_lpr; i++) {
//         sum += p_sorted[i].z;
//         cnt++;
//     }
//     double lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0


//     int init_seeds_num = 0;    // iterate pointcloud, filter those height is less than lpr.height+params_.th_seeds
//     for (int i = 0; i < p_sorted.size(); i++) {
//         if (p_sorted[i].z < lpr_height + th_seed) {
//             init_seeds.push_back(p_sorted[i]);
//         }
//     }
// }

void WildPatchWorkpp::extract_initial_seeds(
        const int zone_idx, const vector<PointXYZ> &p_sorted,
        vector<PointXYZ> &init_seeds) {

    init_seeds.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;

    int init_idx = 0;
    if (zone_idx == 0) {
        for (int i = 0; i < p_sorted.size(); i++) {
            if (p_sorted[i].z < params_.adaptive_seed_selection_margin * params_.sensor_height) {
                ++init_idx;
            } else {
                break;
            }
        }
    }
    
    // Calculate the mean height value.
    for (int i = init_idx; i < p_sorted.size() && cnt < params_.num_lpr; i++) {
        sum += p_sorted[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0

    int init_seeds_num = 0;
    // iterate pointcloud, filter those height is less than lpr.height+params_.th_seeds
    for (int i = 0; i < p_sorted.size(); i++) {
        if (p_sorted[i].z < lpr_height + params_.th_seeds) {
            init_seeds.push_back(p_sorted[i]);
        }
    }
}

void WildPatchWorkpp::reflected_noise_removal(Eigen::MatrixXd &cloud_in)
{
    if ( cloud_in.cols() < 4 ) {
        // cout << "RNR requires intensity information !" << endl;
        // CNOTICE_LOG("RNR requires intensity information !");
        return;
    }

    int cnt = 0;
    for (int i=0; i<cloud_in.rows(); i++)
    {
        double r = sqrt( cloud_in.row(i)(0)*cloud_in.row(i)(0) + cloud_in.row(i)(1)*cloud_in.row(i)(1) );
        double z = cloud_in.row(i)(2);
        double ver_angle_in_deg = atan2(z, r)*180/M_PI;

        if ( ver_angle_in_deg < params_.RNR_ver_angle_thr && z < -params_.sensor_height-0.8 && cloud_in.row(i)(3) < params_.RNR_intensity_thr)
        {
            cloud_nonground_.push_back(PointXYZ(cloud_in.row(i)(0), cloud_in.row(i)(1), cloud_in.row(i)(2), i));
            cloud_in.row(i)(2) = std::numeric_limits<double>::min();
            cnt++;
        }
    }

    // if ( params_.verbose ) CNOTICE_LOG("PatchWorkpp::reflected_noise_removal() - Number of Noises : %d",cnt);
    
    // cout << "PatchWorkpp::reflected_noise_removal() - Number of Noises : " << cnt << endl;
}

void WildPatchWorkpp::addCloud(vector<PointXYZ> &cloud, vector<PointXYZ> &add) {

    cloud.insert(cloud.end(), add.begin(), add.end());
}

void WildPatchWorkpp::addCloud_(vector<PointXYZ> cloud, vector<PointXYZ> &add) {

    cloud.insert(cloud.end(), add.begin(), add.end());
}

double WildPatchWorkpp::calc_point_to_plane_d(PointXYZ p, Eigen::VectorXd normal, double d)
{
    return normal(0)*p.x+normal(1)*p.y+normal(2)*p.z+d;
}


void WildPatchWorkpp::calc_mean_stdev(std::vector<double> vec, double &mean, double &stdev)
{
    if (vec.size() <= 1) return;

    mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();

    for (int i=0; i<vec.size(); i++) { stdev += (vec.at(i)-mean)*(vec.at(i)-mean); }  
    stdev /= vec.size()-1;
    stdev = sqrt(stdev);
}


double WildPatchWorkpp::xy2theta(const double &x, const double &y) { // 0 ~ 2 * PI
    double angle = atan2(y, x);
    return angle > 0 ? angle : 2*M_PI+angle;
}

double WildPatchWorkpp::xy2radius(const double &x, const double &y) {
    // return sqrt(pow(x, 2) + pow(y, 2));
    return sqrt(x*x + y*y);
}

// loc
const double RADIANS_PER_DEGREE = M_PI / 180.0;
const double DEGREES_PER_RADIAN = 180.0 / M_PI;

// WGS84 Parameters
const double WGS84_A = 6378137.0;        // major axis
const double WGS84_B = 6356752.31424518; // minor axis
const double WGS84_F = 0.0033528107;     // ellipsoid flattening
const double WGS84_E = 0.0818191908;     // first eccentricity
const double WGS84_EP = 0.0820944379;    // second eccentricity

// UTM Parameters
const double UTM_K0 = 0.9996;              // scale factor
const double UTM_FE = 500000.0;            // false easting
const double UTM_FN_N = 0.0;               // false northing on north hemisphere
const double UTM_FN_S = 10000000.0;        // false northing on south hemisphere
const double UTM_E2 = (WGS84_E * WGS84_E); // e^2
const double UTM_E4 = (UTM_E2 * UTM_E2);   // e^4
const double UTM_E6 = (UTM_E4 * UTM_E2);   // e^6
const double UTM_EP2 = (UTM_E2 / (1 - UTM_E2)); // e'^2

char WildPatchWorkpp::UTMLetterDesignator(double Lat) {
  char LetterDesignator;

  if ((84 >= Lat) && (Lat >= 72))
    LetterDesignator = 'X';
  else if ((72 > Lat) && (Lat >= 64))
    LetterDesignator = 'W';
  else if ((64 > Lat) && (Lat >= 56))
    LetterDesignator = 'V';
  else if ((56 > Lat) && (Lat >= 48))
    LetterDesignator = 'U';
  else if ((48 > Lat) && (Lat >= 40))
    LetterDesignator = 'T';
  else if ((40 > Lat) && (Lat >= 32))
    LetterDesignator = 'S';
  else if ((32 > Lat) && (Lat >= 24))
    LetterDesignator = 'R';
  else if ((24 > Lat) && (Lat >= 16))
    LetterDesignator = 'Q';
  else if ((16 > Lat) && (Lat >= 8))
    LetterDesignator = 'P';
  else if ((8 > Lat) && (Lat >= 0))
    LetterDesignator = 'N';
  else if ((0 > Lat) && (Lat >= -8))
    LetterDesignator = 'M';
  else if ((-8 > Lat) && (Lat >= -16))
    LetterDesignator = 'L';
  else if ((-16 > Lat) && (Lat >= -24))
    LetterDesignator = 'K';
  else if ((-24 > Lat) && (Lat >= -32))
    LetterDesignator = 'J';
  else if ((-32 > Lat) && (Lat >= -40))
    LetterDesignator = 'H';
  else if ((-40 > Lat) && (Lat >= -48))
    LetterDesignator = 'G';
  else if ((-48 > Lat) && (Lat >= -56))
    LetterDesignator = 'F';
  else if ((-56 > Lat) && (Lat >= -64))
    LetterDesignator = 'E';
  else if ((-64 > Lat) && (Lat >= -72))
    LetterDesignator = 'D';
  else if ((-72 > Lat) && (Lat >= -80))
    LetterDesignator = 'C';
  // 'Z' is an error flag, the Latitude is outside the UTM limits
  else
    LetterDesignator = 'Z';
  return LetterDesignator;
}

void WildPatchWorkpp::LLtoUTM(const double Lat, const double Long,
                               double &UTMNorthing, double &UTMEasting,
                               char *UTMZone) {
  double a = WGS84_A;
  double eccSquared = UTM_E2;
  double k0 = UTM_K0;

  double LongOrigin;
  double eccPrimeSquared;
  double N, T, C, A, M;

  // Make sure the longitude is between -180.00 .. 179.9
  double LongTemp = (Long + 180) - int((Long + 180) / 360) * 360 - 180;

  double LatRad = Lat * RADIANS_PER_DEGREE;
  double LongRad = LongTemp * RADIANS_PER_DEGREE;
  double LongOriginRad;
  int ZoneNumber;

  ZoneNumber = int((LongTemp + 180) / 6) + 1;

  if (Lat >= 56.0 && Lat < 64.0 && LongTemp >= 3.0 && LongTemp < 12.0)
    ZoneNumber = 32;

  // Special zones for Svalbard
  if (Lat >= 72.0 && Lat < 84.0) {
    if (LongTemp >= 0.0 && LongTemp < 9.0)
      ZoneNumber = 31;
    else if (LongTemp >= 9.0 && LongTemp < 21.0)
      ZoneNumber = 33;
    else if (LongTemp >= 21.0 && LongTemp < 33.0)
      ZoneNumber = 35;
    else if (LongTemp >= 33.0 && LongTemp < 42.0)
      ZoneNumber = 37;
  }
  // +3 puts origin in middle of zone
  LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3;
  LongOriginRad = LongOrigin * RADIANS_PER_DEGREE;

  // compute the UTM Zone from the latitude and longitude
  snprintf(UTMZone, 13, "%d%c", ZoneNumber, UTMLetterDesignator(Lat));

  eccPrimeSquared = (eccSquared) / (1 - eccSquared);

  N = a / sqrt(1 - eccSquared * sin(LatRad) * sin(LatRad));
  T = tan(LatRad) * tan(LatRad);
  C = eccPrimeSquared * cos(LatRad) * cos(LatRad);
  A = cos(LatRad) * (LongRad - LongOriginRad);

  M = a *
      ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 -
        5 * eccSquared * eccSquared * eccSquared / 256) *
           LatRad -
       (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32 +
        45 * eccSquared * eccSquared * eccSquared / 1024) *
           sin(2 * LatRad) +
       (15 * eccSquared * eccSquared / 256 +
        45 * eccSquared * eccSquared * eccSquared / 1024) *
           sin(4 * LatRad) -
       (35 * eccSquared * eccSquared * eccSquared / 3072) * sin(6 * LatRad));

  UTMEasting =
      (double)(k0 * N *
                   (A + (1 - T + C) * A * A * A / 6 +
                    (5 - 18 * T + T * T + 72 * C - 58 * eccPrimeSquared) * A *
                        A * A * A * A / 120) +
               500000.0);

  UTMNorthing =
      (double)(k0 * (M + N * tan(LatRad) *
                             (A * A / 2 +
                              (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24 +
                              (61 - 58 * T + T * T + 600 * C -
                               330 * eccPrimeSquared) *
                                  A * A * A * A * A * A / 720)));
  if (Lat < 0)
    UTMNorthing += 10000000.0; // 10000000 meter offset for southern hemisphere
}

void WildPatchWorkpp::LLtoUTM(const double Lat, const double Long,
                               double &UTMNorthing, double &UTMEasting,
                               std::string &UTMZone) {
  char zone_buf[13] = {0};

  LLtoUTM(Lat, Long, UTMNorthing, UTMEasting, zone_buf);

  UTMZone = zone_buf;
}

static double NormalizeAngle(const double angle){
    double a = std::fmod(angle + M_PI,2.0 * M_PI);
    if(a < 0.0){
        a += (2.0 * M_PI);
    }
    return a - M_PI;
}

// void WildPatchWorkpp::HcLocalizationCallback(
//     const serespilot_msgs::msg::Hcinspvatzcb::SharedPtr msg) {
//   double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
//   DataPool* DP = DataPool::Instance();
//   std::unique_lock<std::mutex> locker(loccallback_mutex_);
//   double longitude = msg->longitude;
//   double latitude = msg->latitude;
//   double xg = 0.0;
//   double yg = 0.0;
//   std::string zone;
//   LLtoUTM(latitude, longitude, yg, xg, zone);

//     //   inner_ego_state_.position.x() = xg;
//     //   inner_ego_state_.position.y() = yg;
//     //   inner_ego_state_.lat_acceleration =
//     //       msg->vehicle_linear_acceleration.x * (-1.0);
//     //   inner_ego_state_.lon_acceleration =
//     //       lon_acc_filter_.filter(msg->vehicle_linear_acceleration.y * 9.8);
//     //   inner_ego_state_.lat_velocity = msg->vehicle_linear_velocity.x * (-1.0);
//     //   inner_ego_state_.lon_velocity = msg->vehicle_linear_velocity.y;
//   double heading = (90.0 - msg->heading2) / 180.0 * M_PI;
//   double norm_heading = NormalizeAngle(heading);
//   LocData loc; 
//   loc.time = stamp;
//   loc.xg = xg;
//   loc.yg = yg;
//   loc.heading = heading;
//   loc.norm_heading = norm_heading;
//   CNOTICE_LOG (" =-=====> LOC time: %.10f , norm_heading %10.f , heading %.10f", loc.time,loc.norm_heading,loc.heading);
//             // << " lat: " << latitude 
//             // << ", lon: " << longitude 
//             // << " xg: " << loc.xg
//             // << " yg: " << loc.yg 
//             // << " heading: " << loc.norm_heading
//             // << " deg: " << heading
//             // << std::endl;
//   DP->main_data_.loc_data_cache.Push(loc);
//     //   inner_ego_state_.time_stamp =
//     //       msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
// }

void WildPatchWorkpp::LidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // std::unique_lock<std::mutex> locker(callback_mutex_);
    DataPool* DP = DataPool::Instance();

    const double msg_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    LidarDataInType cached_newest;
    if (DP->main_data_.lidar_data_cache.GetNewest(cached_newest) == 0 &&
        cached_newest.time >= msg_stamp) {
      return;
    }

    std_msgs::msg::Header header;
    header = msg->header;
    LidarDataInType incloud;

    // auto data = msg->data;
    // auto is_dense = msg->is_dense;
    // auto width = msg->width;
    // auto height = msg->height;
    // auto is_bigendian = msg->is_bigendian;
    // auto point_step = msg->point_step;
    // auto row_step = msg->row_step;
    // auto m_time = msg->measurement_time;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // // Convert the sensor_msgs/PointCloud2 message to pcl::PointCloud
    pcl::fromROSMsg(*msg, *cloud);
    // incloud.time = m_time;
    incloud.time = msg_stamp;
    
    // serespilot_msgs::msg::PointXYZIRT *pxyxirt = (serespilot_msgs::msg::PointXYZIRT *)malloc(point_step);
    // size_t point_count = 0;
    // uint32_t point_num = height * width;
    std::cout << std::setprecision(13) <<" =-=====> TOP POINT NUM " << cloud->points.size() << " time: " << incloud.time  << std::endl;

    // 顶雷达 6DoF 标定变换（在回调线程执行，保持轻量）
    // for (size_t i = 0; i < point_num; i++)
    // {
    //     std::memcpy(pxyxirt, data.data() + (i * point_step), point_step);
    //     pcl::PointXYZI point_1;
    //     //RL 下面那条线 右下点
    //     point_1.x = pxyxirt->x;
    //     point_1.y = pxyxirt->y;
    //     point_1.z = pxyxirt->z;
    //     point_1.intensity = pxyxirt->intensity;
    //     cloud->points.push_back(point_1);                //std::cout << "point cloud data: x=" << it.x 
    //     ++point_count;
    // }
    // free(pxyxirt);

    // 顶雷达 6DoF 标定变换
    auto t_trans_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>);
    TransformPointCloud(cloud, cloud_transformed,
                        top_x_offset_, top_y_offset_, top_z_offset_,
                        top_yaw_, top_pitch_, top_roll_);
    auto t_trans_end = std::chrono::high_resolution_clock::now();
    double trans_ms = std::chrono::duration<double, std::milli>(t_trans_end - t_trans_start).count();
    std::cout << "[TopLidar] Transform done:"
              << " pts_in=" << cloud->points.size()
              << " pts_out=" << cloud_transformed->points.size()
              << " offset(x,y,z)=(" << top_x_offset_ << "," << top_y_offset_ << "," << top_z_offset_ << ")"
              << " rpy(yaw,pitch,roll)=(" << top_yaw_ << "," << top_pitch_ << "," << top_roll_ << ")"
              << " body_pitch_deg=" << std::setprecision(3) << (body_pitch_rad_.load() * 57.29577951308232)
              << " cost=" << std::fixed << std::setprecision(3) << trans_ms << "ms" << std::endl;
    incloud.point_cloud_ptr = cloud_transformed;

    DP->main_data_.lidar_data_cache.Push(incloud);
}

// -----------------------------------------------------------------------
// 通用 6DoF 点云刚体变换 (ZYX 欧拉角: yaw->Z, pitch->Y, roll->X)
// -----------------------------------------------------------------------
void WildPatchWorkpp::InsPvaxCallback(const kygi665_msgs::msg::INSPVAX::SharedPtr msg)
{
    if (!msg) {
        return;
    }
    // INSPVAX pitch 单位是度，抬头为正
    constexpr double kDegToRad = 0.017453292519943295;
    body_pitch_rad_.store(msg->pitch * kDegToRad, std::memory_order_relaxed);
}

// /lightning/odom (nav_msgs/Odometry) 回调：从姿态四元数解算机体俯仰角用于点云调平
void WildPatchWorkpp::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!msg) {
        return;
    }
    const auto &q = msg->pose.pose.orientation;
    // 四元数 -> pitch (绕机体 Y 轴)。标准 ROS(X前,Y左,Z上) 下该 pitch 低头为正。
    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    if (sinp > 1.0) sinp = 1.0;
    if (sinp < -1.0) sinp = -1.0;
    const double ros_pitch = std::asin(sinp);
    // body_pitch_rad_ 沿用 INSPVAX 约定：抬头为正，因此取反，调平逻辑保持不变。
    body_pitch_rad_.store(-ros_pitch, std::memory_order_relaxed);
}

void WildPatchWorkpp::TransformPointCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output,
    double x, double y, double z,
    double yaw, double pitch, double roll)
{
    // 安装外参保持不变：雷达坐标系 -> 机体坐标系
    Eigen::Affine3f mount = Eigen::Affine3f::Identity();
    Eigen::AngleAxisf yaw_rot  (static_cast<float>(yaw),   Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf pitch_rot(static_cast<float>(pitch), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf roll_rot (static_cast<float>(roll),  Eigen::Vector3f::UnitX());
    mount.rotate(yaw_rot * pitch_rot * roll_rot);
    mount.translation() << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);

    // 机体俯仰左乘到安装外参上。机体 X 朝前、Z 朝上时，Eigen 的正 Ry 是低头；
    // INSPVAX pitch 抬头为正，因此用 Ry(-pitch) 把点云扳回重力水平。
    const float level_pitch = static_cast<float>(-body_pitch_rad_.load(std::memory_order_relaxed));
    Eigen::Affine3f level = Eigen::Affine3f::Identity();
    level.rotate(Eigen::AngleAxisf(level_pitch, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud(*input, *output, level * mount);
}

// -----------------------------------------------------------------------
// 前雷达回调：接收 lidar_subscribe_front 话题，变换后存入 front_lidar_data_cache
// -----------------------------------------------------------------------
void WildPatchWorkpp::LidarFrontCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    DataPool* DP = DataPool::Instance();

    LidarDataInType incloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    incloud.time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    std::cout << std::setprecision(13) << " =-=====> FRONT POINT NUM " << cloud->points.size()
              << " time: " << incloud.time << std::endl;

    // 前雷达 6DoF 标定变换
    auto t_trans_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>);
    TransformPointCloud(cloud, cloud_transformed,
                        front_x_offset_, front_y_offset_, front_z_offset_,
                        front_yaw_, front_pitch_, front_roll_);
    auto t_trans_end = std::chrono::high_resolution_clock::now();
    double trans_ms = std::chrono::duration<double, std::milli>(t_trans_end - t_trans_start).count();
    std::cout << "[FrontLidar] Transform done:"
              << " pts_in=" << cloud->points.size()
              << " pts_out=" << cloud_transformed->points.size()
              << " offset(x,y,z)=(" << front_x_offset_ << "," << front_y_offset_ << "," << front_z_offset_ << ")"
              << " rpy(yaw,pitch,roll)=(" << front_yaw_ << "," << front_pitch_ << "," << front_roll_ << ")"
              << " cost=" << std::fixed << std::setprecision(3) << trans_ms << "ms" << std::endl;

    incloud.point_cloud_ptr = cloud_transformed;
    DP->main_data_.front_lidar_data_cache.Push(incloud);
}

// -----------------------------------------------------------------------
// 后雷达回调：接收 lidar_subscribe_back 话题，变换后存入 back_lidar_data_cache
// -----------------------------------------------------------------------
void WildPatchWorkpp::LidarBackCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    DataPool* DP = DataPool::Instance();

    LidarDataInType incloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    incloud.time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    std::cout << std::setprecision(13) << " =-=====> BACK POINT NUM " << cloud->points.size()
              << " time: " << incloud.time << std::endl;

    // 后雷达 6DoF 标定变换
    auto t_trans_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>);
    TransformPointCloud(cloud, cloud_transformed,
                        back_x_offset_, back_y_offset_, back_z_offset_,
                        back_yaw_, back_pitch_, back_roll_);
    auto t_trans_end = std::chrono::high_resolution_clock::now();
    double trans_ms = std::chrono::duration<double, std::milli>(t_trans_end - t_trans_start).count();
    std::cout << "[BackLidar] Transform done:"
              << " pts_in=" << cloud->points.size()
              << " pts_out=" << cloud_transformed->points.size()
              << " offset(x,y,z)=(" << back_x_offset_ << "," << back_y_offset_ << "," << back_z_offset_ << ")"
              << " rpy(yaw,pitch,roll)=(" << back_yaw_ << "," << back_pitch_ << "," << back_roll_ << ")"
              << " cost=" << std::fixed << std::setprecision(3) << trans_ms << "ms" << std::endl;

    incloud.point_cloud_ptr = cloud_transformed;
    DP->main_data_.back_lidar_data_cache.Push(incloud);
}

// void WildPatchWorkpp::LidarLeftCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg)
// {
//     std::unique_lock<std::mutex> locker(callback_mutex_);
//     DataPool* DP = DataPool::Instance();

//     // std_msgs::msg::Header header;
//     // header = msg->header;
//     LidarDataInType incloud;

//     auto data = msg->data;
//     auto is_dense = msg->is_dense;
//     auto width = msg->width;
//     auto height = msg->height;
//     auto is_bigendian = msg->is_bigendian;
//     auto point_step = msg->point_step;
//     auto row_step = msg->row_step;
//     auto m_time = msg->measurement_time;

//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
//     // // Convert the sensor_msgs/PointCloud2 message to pcl::PointCloud
//     // pcl::fromROSMsg(*msg, *cloud);
//     incloud.time = m_time;
    
//     serespilot_msgs::msg::PointXYZIRT *pxyxirt = (serespilot_msgs::msg::PointXYZIRT *)malloc(point_step);
//     size_t point_count = 0;
//     uint32_t point_num = height * width;
//     std::cout << std::setprecision(13) <<" =-=====> LEFT POINT NUM " << point_num << "time : " << m_time  << std::endl;

//     sensor_msgs::msg::PointCloud2 cloud_ROS;
//     for (size_t i = 0; i < point_num; i++)
//     {
//         std::memcpy(pxyxirt, data.data() + (i * point_step), point_step);
//         pcl::PointXYZI point_1;
//         //RL 下面那条线 右下点
//         point_1.x = pxyxirt->x;
//         point_1.y = pxyxirt->y;
//         point_1.z = pxyxirt->z;
//         point_1.intensity = pxyxirt->intensity;
//         cloud->points.push_back(point_1);                //std::cout << "point cloud data: x=" << it.x 
//         ++point_count;
//     }
//     free(pxyxirt);
//     incloud.point_cloud_ptr = cloud;
    
//     DP->main_data_.left_lidar_data_cache.Push(incloud);
//     // std::cout << std::setprecision(13)
//     //           << "  ----- WildPatchWorkpp get front left lidar cloud siz : " <<  cloud->points.size() 
//     //           << " | time stamp " << incloud.time
//     //           << std::endl;
//     // std::cout << " DP WildPatchWorkpp GetDataSize() " << DP->main_data_.front_left_lidar_data_cache.GetDataSize() << std::endl;
//     CNOTICE_LOG("  ----- WildPatchWorkpp get front left lidar cloud siz : %d | time stamp %.13d ",cloud->points.size() , incloud.time );
//     CNOTICE_LOG("   DP WildPatchWorkpp GetDataSize() : %d ",DP->main_data_.left_lidar_data_cache.GetDataSize());

// }

// void WildPatchWorkpp::LidarRightCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg)
// {
//     std::unique_lock<std::mutex> locker(callback_mutex_);
//     DataPool* DP = DataPool::Instance();

//     // std_msgs::msg::Header header;
//     // header = msg->header;
//     LidarDataInType incloud;
    
//     auto data = msg->data;
//     auto is_dense = msg->is_dense;
//     auto width = msg->width;
//     auto height = msg->height;
//     auto is_bigendian = msg->is_bigendian;
//     auto point_step = msg->point_step;
//     auto row_step = msg->row_step;
//     auto m_time = msg->measurement_time;

//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
//     // // Convert the sensor_msgs/PointCloud2 message to pcl::PointCloud
//     // pcl::fromROSMsg(*msg, *cloud);
//     incloud.time = m_time;
    
//     serespilot_msgs::msg::PointXYZIRT *pxyxirt = (serespilot_msgs::msg::PointXYZIRT *)malloc(point_step);
//     size_t point_count = 0;
//     uint32_t point_num = height * width;
//     std::cout << std::setprecision(13) <<" =-=====> RIGHT POINT NUM " << point_num << "time : " << m_time  << std::endl;

//     sensor_msgs::msg::PointCloud2 cloud_ROS;
//     for (size_t i = 0; i < point_num; i++)
//     {
//         std::memcpy(pxyxirt, data.data() + (i * point_step), point_step);
//         pcl::PointXYZI point_1;
//         //RL 下面那条线 右下点
//         point_1.x = pxyxirt->x;
//         point_1.y = pxyxirt->y;
//         point_1.z = pxyxirt->z;
//         point_1.intensity = pxyxirt->intensity;
//         cloud->points.push_back(point_1);                //std::cout << "point cloud data: x=" << it.x 
//         ++point_count;
//     }
//     free(pxyxirt);

//     incloud.point_cloud_ptr = cloud;
    
//     DP->main_data_.right_lidar_data_cache.Push(incloud);
//     // std::cout << std::setprecision(13)
//     //           << "  ----- WildPatchWorkpp get front right lidar cloud siz : " <<  cloud->points.size() 
//     //           << " | time stamp " << incloud.time
//     //           << std::endl;
//     // std::cout << " DP WildPatchWorkpp GetDataSize() " << DP->main_data_.front_right_lidar_data_cache.GetDataSize() << std::endl;
//     CNOTICE_LOG("  ----- WildPatchWorkpp get right lidar cloud siz : %d | time stamp %.13d ",cloud->points.size() , incloud.time );
//     CNOTICE_LOG("   DP WildPatchWorkpp GetDataSize() : %d ",DP->main_data_.right_lidar_data_cache.GetDataSize());
    
// }

// void WildPatchWorkpp::LidarFrontCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg)
// {
//     std::unique_lock<std::mutex> locker(callback_mutex_);
//     DataPool* DP = DataPool::Instance();

//     // std_msgs::msg::Header header;
//     // header = msg->header;
//     LidarDataInType incloud;
    
//     auto data = msg->data;
//     auto is_dense = msg->is_dense;
//     auto width = msg->width;
//     auto height = msg->height;
//     auto is_bigendian = msg->is_bigendian;
//     auto point_step = msg->point_step;
//     auto row_step = msg->row_step;
//     auto m_time = msg->measurement_time;

//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
//     // // Convert the sensor_msgs/PointCloud2 message to pcl::PointCloud
//     // pcl::fromROSMsg(*msg, *cloud);
//     incloud.time = m_time;
    
//     serespilot_msgs::msg::PointXYZIRT *pxyxirt = (serespilot_msgs::msg::PointXYZIRT *)malloc(point_step);
//     size_t point_count = 0;
//     uint32_t point_num = height * width;
//     std::cout << std::setprecision(13) <<" =-=====> FRONT POINT NUM " << point_num << "time : " << m_time  << std::endl;

//     sensor_msgs::msg::PointCloud2 cloud_ROS;
//     for (size_t i = 0; i < point_num; i++)
//     {
//         std::memcpy(pxyxirt, data.data() + (i * point_step), point_step);
//         pcl::PointXYZI point_1;
//         //RL 下面那条线 右下点
//         point_1.x = pxyxirt->x;
//         point_1.y = pxyxirt->y;
//         point_1.z = pxyxirt->z;
//         point_1.intensity = pxyxirt->intensity;
//         cloud->points.push_back(point_1);                //std::cout << "point cloud data: x=" << it.x 
//         ++point_count;
//     }
//     free(pxyxirt);

//     incloud.point_cloud_ptr = cloud;
    
//     DP->main_data_.front_lidar_data_cache.Push(incloud);
//     // std::cout << std::setprecision(13)
//     //           << "  ----- WildPatchWorkpp get front right lidar cloud siz : " <<  cloud->points.size() 
//     //           << " | time stamp " << incloud.time
//     //           << std::endl;
//     // std::cout << " DP WildPatchWorkpp GetDataSize() " << DP->main_data_.front_right_lidar_data_cache.GetDataSize() << std::endl;
//     CNOTICE_LOG("  ----- WildPatchWorkpp get front lidar cloud siz : %d | time stamp %.13d ",cloud->points.size() , incloud.time );
//     CNOTICE_LOG("   DP WildPatchWorkpp GetDataSize() : %d ",DP->main_data_.front_lidar_data_cache.GetDataSize());
    
// }

// void WildPatchWorkpp::LidarBackCallback(const serespilot_msgs::msg::PointCloud2::SharedPtr msg)
// {
//     std::unique_lock<std::mutex> locker(callback_mutex_);
//     DataPool* DP = DataPool::Instance();

//     // std_msgs::msg::Header header;
//     // header = msg->header;
//     LidarDataInType incloud;
    
//     auto data = msg->data;
//     auto is_dense = msg->is_dense;
//     auto width = msg->width;
//     auto height = msg->height;
//     auto is_bigendian = msg->is_bigendian;
//     auto point_step = msg->point_step;
//     auto row_step = msg->row_step;
//     auto m_time = msg->measurement_time;

//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
//     // // Convert the sensor_msgs/PointCloud2 message to pcl::PointCloud
//     // pcl::fromROSMsg(*msg, *cloud);
//     incloud.time = m_time;
    
//     serespilot_msgs::msg::PointXYZIRT *pxyxirt = (serespilot_msgs::msg::PointXYZIRT *)malloc(point_step);
//     size_t point_count = 0;
//     uint32_t point_num = height * width;
//     std::cout << std::setprecision(13) <<" =-=====> BACK POINT NUM " << point_num << "time : " << m_time  << std::endl;

//     sensor_msgs::msg::PointCloud2 cloud_ROS;
//     for (size_t i = 0; i < point_num; i++)
//     {
//         std::memcpy(pxyxirt, data.data() + (i * point_step), point_step);
//         pcl::PointXYZI point_1;
//         //RL 下面那条线 右下点
//         point_1.x = pxyxirt->x;
//         point_1.y = pxyxirt->y;
//         point_1.z = pxyxirt->z;
//         point_1.intensity = pxyxirt->intensity;
//         cloud->points.push_back(point_1);                //std::cout << "point cloud data: x=" << it.x 
//         ++point_count;
//     }
//     free(pxyxirt);

//     incloud.point_cloud_ptr = cloud;
    
//     DP->main_data_.back_lidar_data_cache.Push(incloud);
//     // std::cout << std::setprecision(13)
//     //           << "  ----- WildPatchWorkpp get front right lidar cloud siz : " <<  cloud->points.size() 
//     //           << " | time stamp " << incloud.time
//     //           << std::endl;
//     // std::cout << " DP WildPatchWorkpp GetDataSize() " << DP->main_data_.front_right_lidar_data_cache.GetDataSize() << std::endl;
//     CNOTICE_LOG("  ----- WildPatchWorkpp get back lidar cloud siz : %d | time stamp %.13d ",cloud->points.size() , incloud.time );
//     CNOTICE_LOG("   DP WildPatchWorkpp GetDataSize() : %d ",DP->main_data_.back_lidar_data_cache.GetDataSize());
    
// }

void WildPatchWorkpp::Process()
{
    DataPool* DP = DataPool::Instance();

    if(DP->main_data_.lidar_data_cache.GetDataSize() > 0){
        // std::cout << "====== lidar_data_cache in ok "  << DP->main_data_.lidar_data_cache.GetDataSize() << std::endl;
        // 按传感器时间戳取 cache 中最新帧，并丢弃其余帧
        LidarDataInType lidar_raw_data;
        int lidar_discarded = 0;
        DP->main_data_.lidar_data_cache.PopLatestByStamp(
            lidar_raw_data, [](const LidarDataInType &d) { return d.time; }, &lidar_discarded);
        if (lidar_discarded > 0) {
            std::cout << "[Process] lidar_data_cache skipped " << lidar_discarded
                      << " stale frame(s), processing stamp=" << std::setprecision(13)
                      << lidar_raw_data.time << std::endl;
        }
        const double process_lag_ms =
            (node_->get_clock()->now().seconds() - lidar_raw_data.time) * 1000.0;
        std::cout << "[Process] lidar stamp lag vs now=" << std::fixed << std::setprecision(1)
                  << process_lag_ms << " ms" << std::endl;

        // 获取定位数据
        LocData loc_data;
        double threshold_ms = 100;
        if(DP->main_data_.loc_data_cache.GetWithinThreshold(loc_data,lidar_raw_data.time,threshold_ms) < 0){
            std::cout << " ========= CANNOT GET SYNC LOC DATA ===========" << std::endl;
        }else{
            DP->main_data_.loc_data_cache.PopNewest(loc_data);
        }
        // DP->main_data_.loc_data_cache.GetWithinThreshold(loc_data,lidar_raw_data.time,threshold_ms); //先拿最新的数据处理
        // std::cout << " GET loc time " << loc_data.time
        //           << " xg " << loc_data.xg
        //           << " yg " << loc_data.yg
        //           << std::endl;

        bool get_top = true;
        bool get_left_front = false; 
        LidarDataInType front_left_lidar_raw_data;
        if(DP->main_data_.left_lidar_data_cache.GetDataSize() > 0){
            get_left_front = true;
            DP->main_data_.left_lidar_data_cache.PopLatestByStamp(
                front_left_lidar_raw_data, [](const LidarDataInType &d) { return d.time; });
        }
        
        bool get_right_front = false;
        LidarDataInType front_right_lidar_raw_data;
        if(DP->main_data_.right_lidar_data_cache.GetDataSize() > 0){
            get_right_front = true;
            DP->main_data_.right_lidar_data_cache.PopLatestByStamp(
                front_right_lidar_raw_data, [](const LidarDataInType &d) { return d.time; });
        }

        bool get_front = false;
        LidarDataInType front_lidar_raw_data;
        if(DP->main_data_.front_lidar_data_cache.GetDataSize() > 0){
            get_front = true;
            DP->main_data_.front_lidar_data_cache.PopLatestByStamp(
                front_lidar_raw_data, [](const LidarDataInType &d) { return d.time; });
        }

        bool get_back = false;
        LidarDataInType back_lidar_raw_data;
        if(DP->main_data_.back_lidar_data_cache.GetDataSize() > 0){
            get_back = true;
            DP->main_data_.back_lidar_data_cache.PopLatestByStamp(
                back_lidar_raw_data, [](const LidarDataInType &d) { return d.time; });
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr  fuse_cloud_(new pcl::PointCloud<pcl::PointXYZI>);
        // zone fusion for multi lidar 
        
        if(use_multi_lidar_ ){
            //  如果多雷达数据拿到其中一个，则进行融合，否则只用顶雷达数据
            if(get_top || get_left_front || get_right_front || get_back || get_front){

                
                // CNOTICE_LOG(" ===========> GET MULTI LIDAR DATA ");

                // Top
                pcl::PointCloud<PointT>::Ptr top_cloud_raw(new pcl::PointCloud<PointT>);
                top_cloud_raw = lidar_raw_data.point_cloud_ptr;
                for(int i = 0 ; i < top_cloud_raw->points.size() ; i++){
                    const auto& point = top_cloud_raw->points.at(i);
                    if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                        if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                            continue;
                        }

                        // 顶雷达区域滤除
                        // 角度滤除
                        if(fabs(fabs(point.x - top_lidar_x_offset)/fabs(point.y - top_lidar_y_offset))<top_lidar_yaw_offset){
                            continue;
                        }
                        // 距离滤除
                        if(sqrt(pow(point.x - top_lidar_x_offset,2) + pow(point.y - top_lidar_y_offset,2)) < top_lidar_dis_thresh){
                            continue;
                        }
                        PointT pcl_point;
                        pcl_point.x = point.x;
                        pcl_point.y = point.y;
                        pcl_point.z = point.z;
                        pcl_point.intensity = point.intensity;
                        fuse_cloud_->points.push_back(pcl_point);
                    }
                }

                // front
                pcl::PointCloud<PointT>::Ptr front_cloud_raw(new pcl::PointCloud<PointT>);
                front_cloud_raw = front_lidar_raw_data.point_cloud_ptr;
                for(int i = 0 ; i < front_cloud_raw->points.size() ; i++){
                    const auto& point = front_cloud_raw->points.at(i);
                    if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                        if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                            continue;
                        }
                        // 前雷达点云滤除
                        // 角度滤除
                        if(fabs(fabs(point.x - top_lidar_x_offset)/fabs(point.y - top_lidar_y_offset))<top_lidar_yaw_offset){
                            continue;
                        }
                        // 距离滤除
                        if(sqrt(pow(point.x - top_lidar_x_offset,2) + pow(point.y - top_lidar_y_offset,2))  >= top_lidar_dis_thresh){
                            continue;
                        }

                        PointT pcl_point;
                        pcl_point.x = point.x;
                        pcl_point.y = point.y;
                        pcl_point.z = point.z;
                        pcl_point.intensity = point.intensity;
                        fuse_cloud_->points.push_back(pcl_point);
                    }
                }

                // left 
                pcl::PointCloud<PointT>::Ptr left_front_cloud_raw(new pcl::PointCloud<PointT>);
                left_front_cloud_raw = front_left_lidar_raw_data.point_cloud_ptr;
                for(int i = 0 ; i < left_front_cloud_raw->points.size() ; i++){
                    const auto& point = left_front_cloud_raw->points.at(i);
                    if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                        if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                            continue;
                        }
                        // 左雷达点云滤除
                        // 角度滤除,仅考虑在车前和车后的角度
                        if(point.x > car_front){
                            if(fabs(fabs(point.x - top_lidar_x_offset)/fabs(point.y - top_lidar_y_offset)) > left_lidar_yaw_offset){
                                continue;
                            }
                        }
                        // 车后角度与后雷达做比较
                        if(point.x < car_back){
                            if(fabs(fabs(point.x - back_lidar_x_offset)/fabs(point.y - back_lidar_y_offset)) > back_lidar_yaw_offset){
                                continue;
                            }
                        }



                        PointT pcl_point;
                        pcl_point.x = point.x;
                        pcl_point.y = point.y;
                        pcl_point.z = point.z;
                        pcl_point.intensity = point.intensity;
                        fuse_cloud_->points.push_back(pcl_point);
                    }
                }

                // right 
                pcl::PointCloud<PointT>::Ptr right_front_cloud_raw(new pcl::PointCloud<PointT>);
                right_front_cloud_raw = front_right_lidar_raw_data.point_cloud_ptr;
                for(int i = 0 ; i < right_front_cloud_raw->points.size() ; i++){
                    const auto& point = right_front_cloud_raw->points.at(i);
                    if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                        if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                            continue;
                        }
                        // 右雷达点云滤除
                        // 角度滤除,仅考虑在车前和车后的角度
                        if(point.x > car_front){
                            if(fabs(fabs(point.x - top_lidar_x_offset)/fabs(point.y - top_lidar_y_offset)) > right_lidar_yaw_offset){
                                continue;
                            }
                        }
                        // 车后角度与后雷达做比较
                        if(point.x < car_back){
                            if(fabs(fabs(point.x - back_lidar_x_offset)/fabs(point.y - back_lidar_y_offset)) > back_lidar_yaw_offset){
                                continue;
                            }
                        }
                        PointT pcl_point;
                        pcl_point.x = point.x;
                        pcl_point.y = point.y;
                        pcl_point.z = point.z;
                        pcl_point.intensity = point.intensity;
                        fuse_cloud_->points.push_back(pcl_point);
                    }
                }

                // back 
                pcl::PointCloud<PointT>::Ptr back_cloud_raw(new pcl::PointCloud<PointT>);
                back_cloud_raw = back_lidar_raw_data.point_cloud_ptr;
                for(int i = 0 ; i < back_cloud_raw->points.size() ; i++){
                    const auto& point = back_cloud_raw->points.at(i);
                    if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                        if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                            continue;
                        }
                        // 后雷达点云滤除
                        // 角度滤除,仅考虑在车前和车后的角度
                    
                        if(fabs(fabs(point.x - back_lidar_x_offset)/fabs(point.y - back_lidar_y_offset)) < back_lidar_yaw_offset){
                            continue;
                        }
                    
                        PointT pcl_point;
                        pcl_point.x = point.x;
                        pcl_point.y = point.y;
                        pcl_point.z = point.z;
                        pcl_point.intensity = point.intensity;
                        fuse_cloud_->points.push_back(pcl_point);
                    }
                }
            }

        }else{
            std::cout << " ===========> USE ONLY TOP  LIDAR DATA " << std::endl;
            // CNOTICE_LOG(" ===========> USE ONLY TOP  LIDAR DATA ");
            // Top
            pcl::PointCloud<PointT>::Ptr top_cloud_raw(new pcl::PointCloud<PointT>);
            top_cloud_raw = lidar_raw_data.point_cloud_ptr;
            for(int i = 0 ; i < top_cloud_raw->points.size() ; i++){
                const auto& point = top_cloud_raw->points.at(i);
                if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                    if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
                        continue;
                    }
                    PointT pcl_point;
                    pcl_point.x = point.x;
                    pcl_point.y = point.y;
                    pcl_point.z = point.z;
                    pcl_point.intensity = point.intensity;
                    fuse_cloud_->points.push_back(pcl_point);
                }
            }

        }

        
        LidarDataInType fuse_lidar_data;
        fuse_lidar_data.time = lidar_raw_data.time;
        fuse_lidar_data.point_cloud_ptr  = fuse_cloud_;

        pcl::PointCloud<PointT> full_cloud;
        std::vector<int> foreground_idx;

        clock_t t_bef_lidar_proc = clock();
        const int lidar_proc_ret = LidarProcess(loc_data, fuse_lidar_data, full_cloud, foreground_idx);
        clock_t t_aft_lidar_proc = clock();
        std::cout << "[Process] fuse_pts=" << fuse_cloud_->points.size()
                  << " lidar_process=" << (t_aft_lidar_proc - t_bef_lidar_proc) / double(1000000) << "(sec)"
                  << std::endl;
        if (lidar_proc_ret == 0) {
            std::cout << " LidarProcess down! " << std::endl;
            // CNOTICE_LOG( " LidarProcess down! ");
        }

        std::cout << " full_cloud.points.size()" << full_cloud.points.size()
                  << " | foreground_idx.size() " << foreground_idx.size()
                  << std::endl;
        // CNOTICE_LOG(" full_cloud.points.size() : %d | | foreground_idx.size() : %d ",full_cloud.points.size(), foreground_idx.size());

        if(full_cloud.points.size() > 0){
            LidarDataOutType lidar_seg_ret;
            lidar_seg_ret.time = lidar_raw_data.time;
            lidar_seg_ret.full_cloud = full_cloud;
            lidar_seg_ret.foreground_idx = foreground_idx;
            lidar_seg_ret.ground_points_detected = true;
            DP->main_data_.lidar_segs_cache.Push(lidar_seg_ret);
        }
    }

    RetsPub();
}

void WildPatchWorkpp::SpinOnce()
{
    rclcpp::spin_some(node_);
}

void WildPatchWorkpp::RetsPub(){
    // std::unique_lock<std::mutex> locker(publish_mutex_);
    // DataPool* DP = DataPool::Instance();
    // serespilot_msgs::msg::LidarSeg lidar_seg_ret;
    // if(DP->main_data_.lidar_segs_cache.GetDataSize() > 0){
    //     LidarDataOutType cur_seg;
    //     if(DP->main_data_.lidar_segs_cache.PopNewest(cur_seg) > -1){
    //         lidar_seg_ret.header.stamp.sec = trunc(cur_seg.time);
    //         lidar_seg_ret.header.stamp.nanosec = double(cur_seg.time - trunc(cur_seg.time))*1000;

    //         sensor_msgs::msg::PointCloud2 cloud_ROS;
    //         pcl::toROSMsg(cur_seg.full_cloud, cloud_ROS);
    //         lidar_seg_ret.full_cloud = cloud_ROS;

    //         // 
    //         for(auto &idx:cur_seg.foreground_idx){
    //             lidar_seg_ret.foreground_idx.push_back(idx);
    //         }   
            
    //         lidar_seg_ret.ground_points_detected = cur_seg.ground_points_detected;

    //     }
    //     // pub_lidar_seg_ret->publish(lidar_seg_ret);
    //     // std::cout << " -------LIDAR SEG PUB RESTUt : " << cur_seg.time << std::endl;
    //     CNOTICE_LOG(" -------LIDAR SEG PUB RESTUt : %.13d",cur_seg.time);

    // }


}

//data_in 为 current_frame_
int WildPatchWorkpp::LidarProcess(const LocData loc_data,
                                  const LidarDataInType &lidar_raw_data,
                                  pcl::PointCloud<PointT> &full_cloud,
                                  std::vector<int> &foreground_idx) {

    DataPool* DP = DataPool::Instance();
    // pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
    // cloud_raw = lidar_raw_data.point_cloud_ptr;
    // std::cout << " WildPatchWorkpp::LidarProcess cloud_trans size " << cloud_raw->points.size() 
    //             << " | stamp :" << lidar_raw_data.time
    //             << std::endl;
    // // TODO 多雷达点云融合
    
    pcl::PointCloud<PointT> cloud_ = *lidar_raw_data.point_cloud_ptr ;


    // for(int i = 0 ; i < cloud_raw->points.size() ; i++){
    //     const auto& point = cloud_raw->points.at(i);
    //     if ((point.x) < max_x_range_ && (point.x) > min_x_range_ && (point.y) < max_y_range_ && (point.y) > min_y_range_ && point.z < max_z_range_  && point.z > min_z_range_ && !isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
    //         if(point.x < car_front && point.x > car_back && point.y < car_left && point.y > car_right){
    //             continue;
    //         }
    //         PointT pcl_point;
    //         pcl_point.x = point.x;
    //         pcl_point.y = point.y;
    //         pcl_point.z = point.z;
    //         pcl_point.intensity = point.intensity;
    //         // std::cout << " pcl_point.intensity " << pcl_point.intensity
    //         //           << " | same type " << std::is_same<pcl_point.intensity,double>::value
    //         //           << std::endl;
    //         cloud_.push_back(pcl_point);
    //         // std::cout << " X : " << pcl_point.x << std::endl;
    //     }
    // }
    
    // Eigen::MatrixXd cloud_in;
    // cloud_in.resize(cloud_.points.size(), 3);
    // for (size_t i = 0; i < cloud_.points.size(); ++i) {
    //     auto point = cloud_.points.at(i);
    //     cloud_in.row(i) << point.x , point.y , point.z , static_cast<int>(point.intensity) ;
    //     std::cout << "cloud_in.row(i) " <<  cloud_in.row(i) << std::endl;
    // }

    // std::cout << "  ========= TEST cloud_in size " << cloud_.points.size() << std::endl;
    // CNOTICE_LOG("  ========= TEST cloud_in size : %d ",cloud_.points.size());
    // CNOTICE_LOG("  ========= GET loc data time : %.5f , xg : %.5f ,yg : %.5f",loc_data.time,loc_data.xg , loc_data.yg);

    cloud_ground_.clear();
    cloud_nonground_.clear();
    cloud_revert.clear();
    cloud_nonground_new_.clear();

    // if (params_.verbose) cout << "\033[1;32m" << "PatchWorkpp::estimateGround() - Estimation starts !" << "\033[0m" << endl;

    clock_t beg = clock();
    
    // 1. Reflected Noise Removal (RNR)
    // if (params_.enable_RNR) reflected_noise_removal(cloud_in);

    clock_t t1 = clock();

    // 2. Concentric Zone Model (CZM)
    flush_patches(ConcentricZoneModel_);
    flush_patches(ConcentricZoneModel_ground_);
    flush_patches(ConcentricZoneModel_nonground_);

    clock_t t1_1 = clock();

    pc2czm(cloud_, ConcentricZoneModel_);


    // std::shared_ptr<std::vector<Zone>> ConcentricZoneModel_thread;
    // *ConcentricZoneModel_thread = ConcentricZoneModel_;

    // thread process
    // threads.clear();
    // czm_list.clear();
    // for(int i=0; i< threadcount;i++){
    //     czm_list.push_back(ConcentricZoneModel_);
    // }
    // for(size_t i=0; i < threadcount; ++i){
    //     const size_t start = std::floor((i*cloud_in.rows())/threadcount);
    //     const size_t end = std::ceil(((i+1)*cloud_in.rows())/threadcount);
    //     auto &cam_thread = czm_list[i];
    //     threads.push_back(std::thread(&WildPatchWorkpp::pc2czm_thread, this, std::ref(cloud_in), std::ref(cam_thread), start, end));
    // }

    // // wait for results
    // std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    // for (int i = 0; i < ConcentricZoneModel_.size(); i++) {
    //     for(int j = 0; j < ConcentricZoneModel_[i].size();j++ ){
    //         for(int k = 0; k < ConcentricZoneModel_[i][j].size();k++ ){
    //             for(auto &cam_thread:czm_list){
    //                 for(auto &pt:cam_thread[i][j][k]){
    //                     ConcentricZoneModel_[i][j][k].emplace_back(pt);
    //                 }
    //             }
    //         }
    //     }
    // }

    clock_t t2 = clock();

    int concentric_idx = 0;//小圆环ring的idx 最里面为0，计算完一个小圆环就+1

    centers_.clear();
    normals_.clear();

    double t_flush = t1_1-t1, t_czm = t2-t1_1, t_sort = 0.0, t_pca = 0.0, t_gle = 0.0, t_revert = 0.0, t_update = 0.0;

    std::vector<seres::perception::RevertCandidate> candidates;
    std::vector<double> ringwise_flatness;
    

    
    visualization_msgs::msg::MarkerArray mark_array_up;
    int patch_count_ = 0;
    for (int zone_idx = 0; zone_idx < params_.num_zones; ++zone_idx) {
        
        // 一共四个大圆环区域，取其中一个圆环
        auto zone = ConcentricZoneModel_[zone_idx];
        // auto &zone_ground = ConcentricZoneModel_ground_[zone_idx];
        // auto &zone_nonground = ConcentricZoneModel_nonground_[zone_idx];
        // 每个大圆环区域有num_rings_each_zone个小圆环区域ring
        for (int ring_idx = 0; ring_idx < params_.num_rings_each_zone[zone_idx]; ++ring_idx) {
            
            // 每个小圆环区域ring有num_sectors_each_zone个patch，称为sector
            for (int sector_idx = 0; sector_idx < params_.num_sectors_each_zone[zone_idx]; ++sector_idx) {
                
                
                if (zone[ring_idx][sector_idx].size() < params_.num_min_pts)
                {
                    addCloud(cloud_nonground_, zone[ring_idx][sector_idx]);
                    addCloud(ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx],zone[ring_idx][sector_idx]);
                    // std::cout << " 1 " << std::endl;
                    continue;
                }

                patch_count_ += 1 ;
                // --------- region-wise sorting (faster than global sorting method) ---------------- //
                clock_t t_bef_sort = clock();
                // 将区域中的点按从低到高排序，为了方便获得种子点
                sort(zone[ring_idx][sector_idx].begin(), zone[ring_idx][sector_idx].end(), point_z_cmp);
                clock_t t_aft_sort = clock();

                t_sort += t_aft_sort - t_bef_sort;
                // ---------------------------------------------------------------------------------- //

                clock_t t_bef_pca = clock();
                // 初步计算当前区域的地面和非地面
                extract_piecewiseground(zone_idx, zone[ring_idx][sector_idx], regionwise_ground_, regionwise_nonground_);
                // 这个计算完成之后，全局的pc_mean_，normal_参数什么的就是当前patch的
                clock_t t_aft_pca = clock();

                t_pca += t_aft_pca - t_bef_pca;

                centers_.push_back(PointXYZ(pc_mean_(0), pc_mean_(1), pc_mean_(2)));
                normals_.push_back(PointXYZ(normal_(0), normal_(1), normal_(2)));

                clock_t t_bef_gle = clock();
                // Status of each patch
                // used in checking uprightness, elevation, and flatness, respectively
                const double ground_uprightness = normal_(2); // patch在空间中的倾斜程度
                const double ground_elevation   = pc_mean_(2); // patch的高度 
                const double ground_flatness    = singular_values_.minCoeff(); //patch在法向量上的变化程度，可以理解为平坦度
                const double line_variable      = singular_values_(1) != 0 ? singular_values_(0)/singular_values_(1) : std::numeric_limits<double>::max();// 第一个主成分的变化程度与第二个主成分的变化程度之比,可以理解为xy方向上的变换程度之比，比值表明了数据在两个主方向上的伸展差异。该比值越大，数据在主成分方向上的分布越不均匀，意味着数据呈现较强的“线性”或“方向性”，而平面的拟合效果较差。相反，如果比值接近 1，说明数据沿着两个方向的分布更加均匀，接近平面。

                // // 计算近点可通行区域的地面点云高度
                // if(concentric_idx < params_.num_rings_of_interest){
                //     near_ring_ground_count ++;
                //     near_ring_ground_height += ground_elevation;
                // }

                double heading = 0.0;
                for(int i=0; i<3; i++) heading += pc_mean_(i)*normal_(i);
                // 这个heading的物理理解可以表示为，patch平面法向量和（激光雷达原点 - patch平面重心）向量的夹角cos值，一般来说如果是平面或者可以通行的区域，这个夹角一般为钝，那么这个值应该小于0，但对于距离雷达原点较远的patch，这个值的计算算出来可能不准确， 因此在使用这个值的时候，我们只考虑近处的patch

                bool is_upright         = ground_uprightness > params_.uprightness_thr;
                bool is_near_zone       = concentric_idx < params_.num_rings_of_interest;
                // bool is_heading_outside = heading < 0.0;
                bool is_heading_outside = true;

                bool is_not_elevated = false;
                bool is_flat         = false;
                bool is_low_step     = false;

                // visualization_msgs::msg::Marker txt_marker;
                // txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                // txt_marker.action = visualization_msgs::msg::Marker::ADD;
                // txt_marker.header.frame_id = vis_frame_id;
                // txt_marker.ns = "p++_wild";
                // txt_marker.header.stamp.sec = trunc(lidar_raw_data.time);
                // txt_marker.header.stamp.nanosec = static_cast<uint32_t>(
                //     (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
                // // txt_marker.scale.x = 0.5;
                // // txt_marker.scale.y = 0.5;
                // txt_marker.scale.z = 0.6;
                // txt_marker.color.r = 1.0;
                // txt_marker.color.g = 1.0;
                // txt_marker.color.b = 1.0;
                // txt_marker.color.a = 0.9;
                // txt_marker.id = patch_count_;
                // txt_marker.pose.position.x = pc_mean_(0);
                // txt_marker.pose.position.y = pc_mean_(1);
                // txt_marker.pose.position.z = pc_mean_(2);
                // txt_marker.lifetime = rclcpp::Duration::from_seconds(0);
                
                // std::stringstream str;
                // // str << std::setprecision(1)<< "z:"<< zone_idx
                // //                            << "\nr:"<< ring_idx
                // //                            << "\ns:"<< sector_idx;

                // str << std::setprecision(3) << "up:" << ground_uprightness ;
                // txt_marker.text = str.str();
                // mark_array_up.markers.push_back(txt_marker);

                // 按ring的idx来计算
                if (concentric_idx < params_.num_rings_of_interest)
                {
                    is_not_elevated = ground_elevation < params_.elevation_thr[concentric_idx];//初始值为 0 0 0 0 即雷达原点高度
                    is_flat         = ground_flatness < params_.flatness_thr[concentric_idx];
                    // 水平台阶面相对该环地面均值高出不超过 step_height，仍视为地面
                    is_low_step     = ground_elevation < elevation_mean_[concentric_idx] + params_.step_height;
                }
                
                /*
                    Store the elevation & flatness variables
                    for A-GLE (Adaptive Ground Likelihood Estimation)
                    and TGR (Temporal Ground Revert). More information in the paper Patchwork++.
                */
               // 对于近处（ring idx < 4 ）的patch，如果高度不超过阈值（这个阈值是动态变化的，根据前一帧一个ring的所有高度均值来计算的）且倾斜程度不超过阈值，认为是地面
                if (is_upright && is_not_elevated && is_near_zone)
                {
                    update_elevation_[concentric_idx].push_back(ground_elevation);
                    update_flatness_[concentric_idx].push_back(ground_flatness);

                    ringwise_flatness.push_back(ground_flatness); // 这个值会影响近处地面的还原
                }
                // Ground estimation based on conditions
                if (!is_upright) // 倾斜程度剧烈，patch加入非地面
                {
                    addCloud(cloud_nonground_, regionwise_ground_);
                    addCloud(ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx],regionwise_ground_);
                    // std::cout << " 2 " << std::endl;
                }
                else if (!is_near_zone) // 倾斜程度不剧烈 且 不是近处的patch，patch加入地面
                {
                    addCloud(cloud_ground_, regionwise_ground_);
                    addCloud(ConcentricZoneModel_ground_[zone_idx][ring_idx][sector_idx],regionwise_ground_);    
                }
                else if (!is_heading_outside) // 倾斜程度不剧烈 且 是近处的patch 但是heading不对，patch加入非地面
                {
                    addCloud(cloud_nonground_, regionwise_ground_);
                    addCloud(ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx],regionwise_ground_);  
                }
                else if (is_not_elevated || is_flat || is_low_step) // 倾斜程度不剧烈 且 是近处的patch 且 heading正常， 平坦度、均值高度或低矮台阶满足要求则 patch加入地面
                {
                    addCloud(cloud_ground_, regionwise_ground_);
                    addCloud(ConcentricZoneModel_ground_[zone_idx][ring_idx][sector_idx],regionwise_ground_); 
                }
                else // 倾斜程度不剧烈 ，且是近处的patch，且heading正常，但平坦度和均值高度没有满足需求，加入二次判断候选，待定
                {
                    seres::perception::RevertCandidate candidate(concentric_idx, sector_idx, ground_flatness, line_variable, pc_mean_, regionwise_ground_);
                    candidates.push_back(candidate);
                }
                // Every regionwise_nonground is considered nonground.
                addCloud(cloud_nonground_, regionwise_nonground_);
                addCloud(ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx],regionwise_nonground_); 
                // std::cout << " 3 " << std::endl;
                clock_t t_aft_gle = clock();

                t_gle += t_aft_gle - t_bef_gle;

                // std::cout << " **** zone_nonground[ring_idx][sector_idx].size() " << ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx].size() << std::endl;



            }

            clock_t t_bef_revert = clock();
            
            // 把一个ring计算完成之后，才会进行二次候选的判断计算，这个判断其实是利用的ring的平坦度计算值
            if (!candidates.empty())
            {
                if (params_.enable_TGR) {
                    temporal_ground_revert(ringwise_flatness, candidates, concentric_idx);
                } else {
                    for ( auto candidate : candidates ) {
                        addCloud(cloud_nonground_, candidate.regionwise_ground);


                    }
                }

                candidates.clear();
                ringwise_flatness.clear();
            }
            clock_t t_aft_revert = clock();
            
            t_revert += t_aft_revert - t_bef_revert;

            concentric_idx++;
        }
    }


    std::vector<PointXYZ> my_revert_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> patch_viz_cloud;
    visualization_msgs::msg::MarkerArray mark_array;
    mark_array.markers.clear();
    clock_t t_bef_mrt = clock();
    int mrt_patch_cnt = 0;
    int mrt_no_nb_cnt = 0;
    if(params_.enable_MRT){
        //MY REVERT
        if (params_.verbose) {
            std::cout << "START MY REVERT " << std::endl;
        }
        // CNOTICE_LOG(" START MY REVERT ");

        cloud_nonground_.clear();
        // 迭代多次？
        // for(int iter=0 ; iter < 2 ;  ++iter){
            // std::cout << " MY REVERT ITER " << iter << std::endl;
            int patch_count = -1;
            int abs_ring_idx = -1; 
            double near_ring_ground_height = 0;
            int near_ring_ground_count = 0;
            for (int zone_idx = 0; zone_idx < params_.num_zones; ++zone_idx) {
                // 一共四个大圆环区域，取其中一个圆环
                auto zone_ground = ConcentricZoneModel_ground_[zone_idx];
                auto zone_nonground = ConcentricZoneModel_nonground_[zone_idx];
                // 每个大圆环区域有num_rings_each_zone个小圆环区域ring
                for (int ring_idx = 0; ring_idx < params_.num_rings_each_zone[zone_idx]; ++ring_idx) {
                    abs_ring_idx += 1;
                    // 每个小圆环区域ring有num_sectors_each_zone个patch，称为sector
                    for (int sector_idx = 0; sector_idx < params_.num_sectors_each_zone[zone_idx]; ++sector_idx) {
                        patch_count += 1;
                        // std::cout << "zone_nonground[ring_idx][sector_idx].size() " << zone_nonground[ring_idx][sector_idx].size() << std::endl;
                        // std::cout << "-zone_ground[ring_idx][sector_idx].size() " << zone_ground[ring_idx][sector_idx].size() << std::endl;
                        if (zone_nonground[ring_idx][sector_idx].size() < params_.num_min_pts)
                        {
                            continue;
                        }
                        mrt_patch_cnt++;
                        // 计算当前patch非地面点的重心
                        //self non ground center
                        Eigen::Vector3d non_center;
                        Eigen::MatrixX3d eigen_non_ground(zone_nonground[ring_idx][sector_idx].size(), 3);
                        int j=0;
                        for (auto &p: zone_nonground[ring_idx][sector_idx]) {
                            eigen_non_ground.row(j++) << p.x, p.y, p.z;
                        }
                        
                        non_center << eigen_non_ground.colwise().mean()(0), eigen_non_ground.colwise().mean()(1), eigen_non_ground.colwise().mean()(2);

                        /*
                        [1 ][2 ][3 ]
                        [4 ][* ][5 ]
                        [6 ][7 ][8 ]
                        */
                        //计算相邻patch地面点中xy平面距离最近的是哪一个
                        //计算地面点均值重心,8邻域
                        double  min_dist = std::numeric_limits<double>::max();
                        int closet_nb_zone_idx = -1;
                        int closet_nb_ring_idx = -1;
                        int closet_nb_sector_idx = -1;

                        
                        std::vector<double> closet_dist_vec;
                        // if (params_.verbose) {
                        //     std::cout << " CURRENT NON GROUND IDX" << zone_idx << " / " << ring_idx << " / " << sector_idx << std::endl;
                        // }
                        // CNOTICE_LOG(" CURRENT NON GROUND IDX : %d  / %d  / %d ", zone_idx,ring_idx,sector_idx );
                        int neighbour_cal_count = 0;
                        for(auto nb:ConcentricZoneModel_neighbour_[zone_idx][ring_idx][sector_idx]){
                            int nb_zone_idx = nb.zone_idx;
                            int nb_ring_idx = nb.ring_idx;
                            int nb_sector_idx = nb.sector_idx;
                            std::vector<PointXYZ> nb_patch_ground = ConcentricZoneModel_ground_[nb_zone_idx][nb_ring_idx][nb_sector_idx];
                            // std::cout << " cal nb " << nb_zone_idx << " / " << nb_ring_idx << " / " << nb_sector_idx 
                            //           << " nb_patch_ground size " << nb_patch_ground.size()  
                            //           << std::endl;
                            std::vector<double> dis_vec; 
                            if(nb_patch_ground.size() > 0){
                                Eigen::MatrixX3d eigen_ground(nb_patch_ground.size(), 3);
                                int j=0;
                                for (auto p: nb_patch_ground) {
                                    eigen_ground.row(j++) << p.x, p.y, p.z;
                                    double cur_dis = sqrt(pow(non_center[0] - p.x,2)+pow(non_center[1] - p.y,2)+pow(non_center[2] - p.z,2));
                                    dis_vec.push_back(cur_dis);  
                                    // std::cout << " p : " << p.x << " " << p.y << " " << p.z 
                                    //           << " | dist " << cur_dis << std::endl;
                                }
                                // 计算哪个接壤点离现在的非地面点重心最近
                                double cur_patch_min_dist = *(std::min_element(dis_vec.begin(),dis_vec.end()));
                                
                                if(cur_patch_min_dist < min_dist){
                                    min_dist = cur_patch_min_dist;
                                    closet_nb_zone_idx = nb_zone_idx;
                                    closet_nb_ring_idx = nb_ring_idx;
                                    closet_nb_sector_idx = nb_sector_idx;
                                    closet_dist_vec = dis_vec;
                                }

                                // std::cout << " | cur_patch_min_dist " << cur_patch_min_dist << " MIN : " << min_dist << std::endl;
                                neighbour_cal_count += 1;
                            }
                        }
                        

                        // std::cout << "==> neighbour_cal_count " << neighbour_cal_count  <<  std::endl;
                        if( closet_nb_zone_idx > -1 && closet_nb_ring_idx > -1 && closet_nb_sector_idx > -1 ){
                            // std::cout << "==> closet idx " << closet_nb_zone_idx << "/" << closet_nb_ring_idx << "/" << closet_nb_sector_idx << std::endl;
                            // 计算最近的patch ground内在range范围内的点以及算出最近点
                            auto nearest_nb_patch_ground = ConcentricZoneModel_ground_[closet_nb_zone_idx][closet_nb_ring_idx][closet_nb_sector_idx];
                            Eigen::Vector3d closet_patch_point_;
                            Eigen::MatrixXd eigen_closet_patch_cal;
                            std::vector<PointXYZ> cal_patch_ground_in_range;
                            for(int i=0 ;i<nearest_nb_patch_ground.size();i++){
                                if( zone_idx < 2 && closet_dist_vec[i] < min_dist + params_.near_cal_range){
                                    cal_patch_ground_in_range.push_back(nearest_nb_patch_ground[i]);
                                }else if(zone_idx >= 2 && closet_dist_vec[i] < min_dist + params_.far_cal_range){
                                    cal_patch_ground_in_range.push_back(nearest_nb_patch_ground[i]);
                                }
                                if(abs(closet_dist_vec[i] - min_dist) < 0.1){ // 最近点坐标
                                    closet_patch_point_[0] = nearest_nb_patch_ground[i].x;
                                    closet_patch_point_[1] = nearest_nb_patch_ground[i].y;
                                    closet_patch_point_[2] = nearest_nb_patch_ground[i].z;
                                }
                            }
                            
                            eigen_closet_patch_cal.resize(cal_patch_ground_in_range.size(),3);
                            j=0;
                            for(auto &p:cal_patch_ground_in_range){
                                eigen_closet_patch_cal.row(j++) << p.x , p.y , p.z ;
                            }

                            Eigen::Vector3d closet_patch_cal_center;
                            closet_patch_cal_center << eigen_closet_patch_cal.colwise().mean()(0), eigen_closet_patch_cal.colwise().mean()(1), eigen_closet_patch_cal.colwise().mean()(2);

                            // std::cout << " CLOSE DIST " << min_dist  << " \n" << closet_patch_point_
                            //         << " c:" <<  zone_idx << "/" <<ring_idx << "/" << sector_idx
                            //         << "\ncloset:" << closet_nb_zone_idx << "/" <<closet_nb_ring_idx << "/" << closet_nb_sector_idx << std::endl;

                            // CNOTICE_LOG(" CLOSE DIST : %d \n %", min_dist );

                            // 判断当前patch非地面点能否恢复成地面点
                            std::vector<PointXYZ> tmp_ground;
                            std::vector<PointXYZ> tmp_nonground;
                            auto &patch_pts = zone_nonground[ring_idx][sector_idx];
                            if (cal_patch_ground_in_range.size() < 3) {
                                addCloud(cloud_nonground_, patch_pts);
                            } else {
                            estimate_plane(cal_patch_ground_in_range); //
                            // 记录相对邻接地面的高度。用 90% 分位代替最高点，避免个别噪点否决整块台阶
                            std::vector<double> distances;
                            distances.reserve(patch_pts.size());
                            double max_distance = 0.0;
                            for (const auto &p : patch_pts) {
                                double distance = calc_point_to_plane_d(p, normal_, d_);
                                distances.push_back(distance);
                                if (distance > max_distance) {
                                    max_distance = distance;
                                }
                            }
                            double robust_height = max_distance;
                            if (distances.size() >= 2) {
                                std::vector<double> sorted_dist = distances;
                                const size_t p90_idx = static_cast<size_t>(0.9 * (sorted_dist.size() - 1));
                                std::nth_element(sorted_dist.begin(), sorted_dist.begin() + p90_idx, sorted_dist.end());
                                robust_height = sorted_dist[p90_idx];
                            }
                            const double step_height = std::max(params_.step_height, params_.obstacle_height);
                            // 绝大多数点都贴在台阶高度以内，说明这是低矮台阶而不是竖立障碍
                            const bool low_step_patch = robust_height < step_height;

                            double ring_revert_thresh = params_.far_revert_thresh;
                            if (abs_ring_idx <= 6 && abs_ring_idx >= 0) {
                                ring_revert_thresh = params_.near_revert_thresh;
                            } else if (abs_ring_idx <= 11) {
                                ring_revert_thresh = params_.middle_revert_thresh;
                            }

                            if (max_distance < params_.obstacle_height) {
                                // 最高点低于 obstacle_height，整块认定为地面
                                addCloud(cloud_ground_, patch_pts);
                                addCloud(tmp_ground, patch_pts);
                                addCloud(ConcentricZoneModel_ground_[zone_idx][ring_idx][sector_idx], patch_pts);
                                addCloud(my_revert_cloud, tmp_ground);
                            } else {
                                for (size_t pi = 0; pi < patch_pts.size(); ++pi) {
                                    const double distance = distances[pi];
                                    // 低矮台阶收回台阶面；更高的尖峰仍保留为障碍。其余 patch 沿用原距离阈值
                                    const bool revert_as_step = low_step_patch && distance < step_height;
                                    const bool revert_as_near = std::abs(distance) < ring_revert_thresh;
                                    if (revert_as_step || revert_as_near) {
                                        cloud_ground_.push_back(patch_pts[pi]);
                                        tmp_ground.push_back(patch_pts[pi]);
                                        ConcentricZoneModel_ground_[zone_idx][ring_idx][sector_idx].push_back(patch_pts[pi]);
                                    } else {
                                        cloud_nonground_.push_back(patch_pts[pi]);
                                        tmp_nonground.push_back(patch_pts[pi]);
                                    }
                                }
                                addCloud(my_revert_cloud, tmp_ground);
                            }
                            }


                            

                            if(params_.verbose){ // 可视化 patch非地面点
                                srand(patch_count);
                                int r = rand()%256;
                                int g = rand()%256;
                                int b = rand()%256;

                                // int r_ = rand()%50;
                                // int g_ = rand()%50;
                                // int b_ = rand()%50;
                                for(int i=0;i<zone_nonground[ring_idx][sector_idx].size();i++){
                                    pcl::PointXYZRGB point;
                                    point.x = zone_nonground[ring_idx][sector_idx][i].x;
                                    point.y = zone_nonground[ring_idx][sector_idx][i].y;
                                    point.z = zone_nonground[ring_idx][sector_idx][i].z;
                                    point.r = 255;
                                    point.g = 0;
                                    point.b = 0;
                                    patch_viz_cloud.push_back(point);
                                }
                                // for(int i=0;i<zone_ground[ring_idx][sector_idx].size();i++){
                                //     pcl::PointXYZRGB point;
                                //     point.x = zone_ground[ring_idx][sector_idx][i].x;
                                //     point.y = zone_ground[ring_idx][sector_idx][i].y;
                                //     point.z = zone_ground[ring_idx][sector_idx][i].z;
                                //     point.r = 150 - r_;
                                //     point.g = 150 - g_;
                                //     point.b = 150 - b_;
                                //     patch_viz_cloud.push_back(point);
                                // }
                                // std::cout << " patch_viz_cloud size " << patch_viz_cloud.points.size() << std::endl;
                                // add neighbour patch up zone idx 
                                
                                    
                                pcl::PointXYZRGB point_;
                                point_.x = closet_patch_point_[0];
                                point_.y = closet_patch_point_[1];
                                point_.z = closet_patch_point_[2];
                                point_.r = 255;
                                point_.g = 255;
                                point_.b = 255;
                                patch_viz_cloud.push_back(point_);

                                point_.x = non_center[0];
                                point_.y = non_center[1];
                                point_.z = non_center[2];
                                point_.r = 0;
                                point_.g = 255;
                                point_.b = 0;
                                patch_viz_cloud.push_back(point_);
                            
                            
                                for(int i=0;i<cal_patch_ground_in_range.size();i++){
                                    if(abs(cal_patch_ground_in_range[i].x -closet_patch_point_[0]) < 0.1 && 
                                      abs(cal_patch_ground_in_range[i].y -closet_patch_point_[1]) < 0.1 &&
                                      abs(cal_patch_ground_in_range[i].z -closet_patch_point_[2]) < 0.1) 
                                      continue;
                                    pcl::PointXYZRGB point;
                                    point.x = cal_patch_ground_in_range[i].x;
                                    point.y = cal_patch_ground_in_range[i].y;
                                    point.z = cal_patch_ground_in_range[i].z;
                                    point.r = 0;
                                    point.g = 0;
                                    point.b = 255;
                                    // point.r = 0.7*r;
                                    // point.g = 0.7*g;
                                    // point.b = 0.7*b;
                                    patch_viz_cloud.push_back(point);
                                }
                                

                                



                                Eigen::MatrixX3d eigen_ground(zone_nonground[ring_idx][sector_idx].size(), 3);
                                int j=0;
                                for (auto &p: zone_nonground[ring_idx][sector_idx]) {
                                    eigen_ground.row(j++) << p.x, p.y, p.z;
                                }
                                Eigen::MatrixX3d centered = eigen_ground.rowwise() - eigen_ground.colwise().mean();
                                Eigen::VectorXd non_center ;
                                non_center.resize(3);
                                non_center << eigen_ground.colwise().mean()(0), eigen_ground.colwise().mean()(1), eigen_ground.colwise().mean()(2);

                                visualization_msgs::msg::Marker txt_marker;
                                txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                                txt_marker.action = visualization_msgs::msg::Marker::ADD;
                                txt_marker.header.frame_id = vis_frame_id;
                                txt_marker.ns = "p++_wild";
                                txt_marker.header.stamp.sec = trunc(lidar_raw_data.time);
                                txt_marker.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
                                // txt_marker.scale.x = 0.5;
                                // txt_marker.scale.y = 0.5;
                                txt_marker.scale.z = 0.6;
                                txt_marker.color.r = 1.0;
                                txt_marker.color.g = 1.0;
                                txt_marker.color.b = 1.0;
                                txt_marker.color.a = 0.9;
                                txt_marker.id = patch_count;
                                txt_marker.pose.position.x = non_center(0);
                                txt_marker.pose.position.y = non_center(1);
                                txt_marker.pose.position.z = non_center(2);
                                txt_marker.lifetime = rclcpp::Duration::from_seconds(0);
                                
                                std::stringstream str;
                                // str << std::setprecision(1)<< "z:"<< zone_idx
                                //                            << "\nr:"<< ring_idx
                                //                            << "\ns:"<< sector_idx;

                                // str << std::setprecision(1) << "c:" <<  zone_idx << "/" <<ring_idx << "/" << sector_idx
                                //                             << "\nnb"  << closet_nb_zone_idx << "/" << closet_nb_ring_idx << "/" << closet_nb_sector_idx;
                                txt_marker.text = str.str();
                                mark_array.markers.push_back(txt_marker);
                            }
                        }else{
                            mrt_no_nb_cnt++;
                            // if (params_.verbose) {
                            //     std::cout << " NO CLOSE NB GOUNRD "
                            //               << zone_idx << "/" << ring_idx << "/" << sector_idx
                            //               << std::endl;
                            // }
                            addCloud(cloud_nonground_,ConcentricZoneModel_nonground_[zone_idx][ring_idx][sector_idx]);
                            
                        }

                    }
                }
            }
        // }
        
    }
    clock_t t_aft_mrt = clock();
    const double t_mrt = t_aft_mrt - t_bef_mrt;

    // if(near_ring_ground_count){
    //     std::cout << "\033[1;32m" << " =========== MODI : near_ring_ground_count " << near_ring_ground_count 
    //               << "\033[0m"  << std::endl;
    //     std::cout << "\033[1;32m" << " =========== MODI : near_ring_ground_height "  << near_ring_ground_height / near_ring_ground_count
    //               << "\033[0m" << std::endl;
    // }
    // std::cout << " Current elevation_thr " << params_.elevation_thr[0] << " | "  
    //                                        << params_.elevation_thr[1] << " | "  
    //                                        << params_.elevation_thr[2] << " | "  
    //                                        << params_.elevation_thr[3] << " | "  
    //                                        << std::endl;
    clock_t t_bef_update = clock();
    update_elevation_thr();
    update_flatness_thr();
    clock_t t_aft_update = clock();

    t_update = t_aft_update - t_bef_update;

    // 简单噪点滤除，半径过滤
    clock_t t_bef_filter = clock();
    if(use_filter_){
        vector<PointXYZ> tmp_cloud_nonground_;
        // cloud_nonground_.clear();
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        for(int i=0;i<cloud_nonground_.size();i++){
            if(cloud_nonground_[i].x > filter_front_range_ || 
               fabs(cloud_nonground_[i].y) > filter_side_range_ || 
               cloud_nonground_[i].x < filter_back_range_ ){
                tmp_cloud_nonground_.push_back(PointXYZ( cloud_nonground_[i].x,
                                                         cloud_nonground_[i].y, 
                                                         cloud_nonground_[i].z, 
                                                         -1 ,
                                                         cloud_nonground_[i].intensity));
                continue;
            }
            pcl::PointXYZI point;
            point.x = cloud_nonground_[i].x;
            point.y = cloud_nonground_[i].y;
            point.z = cloud_nonground_[i].z;
            point.intensity = cloud_nonground_[i].intensity;
            cloud_filter->points.emplace_back(point);
        }
        if(cloud_filter->points.size() > 1){
            
            pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;	//创建滤波器对象
            ror.setInputCloud(cloud_filter);						//设置待滤波点云
            ror.setRadiusSearch(filter_nb_radius_);						//设置查询点的半径范围
            ror.setMinNeighborsInRadius(filter_nb_num_);	//设置判断是否为离群点的阈值，即半径内至少包括的点数
            
            ror.filter(*cloud_filtered);	//执行滤波，保存滤波结果于cloud_filtered

            
            for(const auto &pts:cloud_filtered->points){
                // PointXYZ pt;
                // pt.x = pts.x;
                // pt.y = pts.y;
                // pt.z = pts.z;
                // pt.intensity = pts.intensity;
                tmp_cloud_nonground_.push_back(PointXYZ( pts.x, pts.y, pts.z, -1 ,pts.intensity));
            }

        }else{
            std::cout << " NO POINTS IN RANGE FOR FILTER " << std::endl;
        }
        cloud_nonground_.clear();
        cloud_nonground_ = tmp_cloud_nonground_;
    }

    clock_t t_aft_filter = clock();
    double t_filter = t_aft_filter - t_bef_filter;



    clock_t t_bef_convert = clock();

    pcl::PointCloud<PointT> convert_cloud,cur_cloud_to_trans_cloud;
    std::cout.precision(17);
    
    if(use_loc_trans_){
        // 将前期检测到的非地面点转换到盲区内，进行记忆
        // 1. 历史盲区点列表
        // 2. 待转换区域范围就使用滤波的范围作为记忆范围，盲区范围为自车范围

        // step 1 ：
        // 提取当前帧的待转换区域范围内的点填充
        
        vector<PointXYZ> cur_cloud_to_trans,cur_cloud_in_blind_rel,cur_cloud_in_blind;
        for(int i=0;i<cloud_nonground_.size();i++){
            if (cloud_nonground_[i].x > filter_back_range_ && 
                cloud_nonground_[i].x < filter_front_range_  && 
                fabs(cloud_nonground_[i].y) < filter_side_range_ ){ //car
                
                // PointT point_;
                // point_.x = cloud_nonground_[i].x;
                // point_.y = cloud_nonground_[i].y;
                // point_.z = cloud_nonground_[i].z;
                // point_.intensity = cloud_nonground_[i].intensity;
                // std::cout << " cur_cloud_to_trans_cloud  " << point_.x << " | " << point_.y << std::endl;
                // cur_cloud_to_trans_cloud.points.push_back(point_);
                // 因为已经去除过一次盲区的点云，所以现在的点就是待记忆区域范围内除盲区之外的点云了
                // 将点云坐标转换为全局坐标
                double g_x = (cos(loc_data.norm_heading) * cloud_nonground_[i].x) +
                      ((-1.0) * sin(loc_data.norm_heading) * cloud_nonground_[i].y);
                
                double g_y = (sin(loc_data.norm_heading) * cloud_nonground_[i].x) +
                      (cos(loc_data.norm_heading) * cloud_nonground_[i].y);
                
                g_x += loc_data.xg;
                
                g_y += loc_data.yg;
                // std::cout << " OBSTACLE abs " << g_x << " | " << g_y << std::endl; 
                // rot_p.heading = MathUtil::NormalizeAngle(abs_base.heading + rel.heading);
                cur_cloud_to_trans.push_back(PointXYZ( g_x , 
                                                       g_y, 
                                                       cloud_nonground_[i].z, 
                                                       -1 ,
                                                       cloud_nonground_[i].intensity ));

            }
            
        }
        std::cout << " cur_cloud_to_trans fill in " << cur_cloud_to_trans.size() << std::endl;
        std::cout << " last_cloud_to_trans size " << last_cloud_to_trans.size() << std::endl;

        //step 2： 判断上一帧的待转换列表是否有数据，有的话进行转换，没有的话填充当前帧待转换点云至上一帧待转换点云，等待下一帧的时候使用
        
        if(last_cloud_to_trans.size() < 1 ){
            // size小于1 可能有两种情况，初始化的第一帧，或者待转换区域范围内就没有待转换的点云
            std::cout << " LAST FRAME has no cloud to trans , fill in" << std::endl;
            for(const auto &pts:cur_cloud_to_trans){
                last_cloud_to_trans.push_back(pts);
            }    
            // // 盲区点云也保存至上一帧待转换点云等待使用
            // for(const auto&pts:last_cloud_in_blind){
            //     last_cloud_to_trans.push_back(pts);
            // }

        }else{// 存在待转换点云，等待转换
            std::cout << " LAST FRAME has cloud to trans " << std::endl;
            for(int i=0;i<last_cloud_to_trans.size();i++){
                // 将全局坐标转换至当前坐标，判断是否在盲区内
                // std::cout << " OBSTACLE abs " << last_cloud_to_trans[i].x << " | " << last_cloud_to_trans[i].y << std::endl; 
                double x = last_cloud_to_trans[i].x  - loc_data.xg;
                double y = last_cloud_to_trans[i].y  - loc_data.yg;

                double rel_x = (cos(loc_data.norm_heading) * x) + 
                               (sin(loc_data.norm_heading) * y);
                double rel_y = ((-1.0) * sin(loc_data.norm_heading) * x) + 
                               (cos(loc_data.norm_heading) * y);
                // std::cout << " OBSTACLE  " << rel_x << " | " << rel_y << std::endl; 
                PointT point_;
                point_.x = rel_x;
                point_.y = rel_y;
                point_.z = last_cloud_to_trans[i].z;
                point_.intensity = last_cloud_to_trans[i].intensity;
                convert_cloud.points.push_back(point_);
                // rot_p.heading = MathUtil::NormalizeAngle(abs.heading - abs_base.heading);
                if(rel_x < convert_front_range_ && rel_x > car_back && rel_y > car_right && rel_y < car_left){
                    // std::cout << " OBSTACLE IN BLIND " << rel_x << " | " << rel_y << std::endl; 
                    // 相对的障碍物坐标输出
                    cur_cloud_in_blind_rel.push_back(PointXYZ( rel_x , 
                                                       rel_y, 
                                                       last_cloud_to_trans[i].z, 
                                                       -1 ,
                                                       last_cloud_to_trans[i].intensity ));
                    // 绝对的障碍物坐标就记录下来，下一帧使用
                    cur_cloud_in_blind.push_back(PointXYZ( last_cloud_to_trans[i].x  , 
                                                       last_cloud_to_trans[i].y, 
                                                       last_cloud_to_trans[i].z, 
                                                       -1 ,
                                                       last_cloud_to_trans[i].intensity ));
                }
            }


        }
        // 将当前帧的待转换区域点云和盲区点云填充，等待下一帧使用
        last_cloud_to_trans.clear();
        convert_count ++ ;
        for(const auto& pts:cur_cloud_in_blind){
            last_cloud_to_trans.push_back(pts);
        }
        for(const auto &pts:cur_cloud_to_trans){
            last_cloud_to_trans.push_back(pts);
        }    
        // 每100帧清除一次补盲区域的转换点，避免动态障碍物的干扰
        if(use_loc_trans_remove_ && convert_count > convert_count_thresh_){
            std::cout << " ------------------------------------------------------------ REMOVE " << std::endl;
            last_cloud_to_trans.clear();
            convert_count = 0;
        }

        //step 3 将当前盲区内的点云添加至当前帧非地面点
        for(const auto& pts:cur_cloud_in_blind_rel){
            // std::cout << " car front " << car_front << std::endl;
            if(pts.y > car_right && pts.y <car_left && pts.x > car_back && pts.x < car_front){// 只输出自车范围内的记忆坐标
                if(fabs(pts.y) > convert_blind_side_range_ || pts.x > convert_blind_front_range_  ){
                    // continue;
                    cloud_nonground_.push_back(PointXYZ(pts.x, pts.y, pts.z, -1,pts.intensity));
                
                    
                }
                // cloud_nonground_.push_back(PointXYZ(pts.x, pts.y, pts.z, -1,pts.intensity));
            }
            
            // cloud_nonground_.push_back(PointXYZ(pts.x, pts.y, pts.z, -1,pts.intensity));
        }

    }
    clock_t t_aft_convert = clock();
    double t_convert = t_aft_convert - t_bef_convert;

    clock_t t_bef_detect = clock();
    pcl::PointCloud<PointT> tmp_cloud_ncr;
    if(use_lidar_detect_){
        // normal change rate 
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_normal(new pcl::PointCloud<pcl::PointXYZ>);
        for(int i=0;i<cloud_nonground_.size();i++){
                pcl::PointXYZ point;
                point.x = cloud_nonground_[i].x;
                point.y = cloud_nonground_[i].y;
                point.z = cloud_nonground_[i].z;
                // point.intensity = cloud_nonground_[i].intensity;
                cloud_normal->points.emplace_back(point);
        }
    
        std::vector<float> ncr_results(cloud_normal->points.size(), 0.0f);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud_normal);
        
        pcl::PCA<pcl::PointXYZ> pca;
        // pca.setInputCloud(cloud);

        // 获取协方差矩阵的特征值
        
        // 遍历每个点计算NCR
        // #pragma omp parallel for
        // double search_radius = 0.3;
        
        for (size_t i = 0; i < cloud_normal->points.size(); ++i)
        {
            const auto& pt = cloud_normal->points[i];
            std::vector<int> neighbor_indices;
            std::vector<float> distances;

            // 执行半径搜索
            double nb_num = kdtree.radiusSearch(pt, filter_nb_radius_, neighbor_indices, distances);
            // std::cout << " nb_num " << nb_num << std::endl;
            if (nb_num > detect_num_)
            {
                float sum_diff = 0.0f;
                int valid_neighbors = 0;

                // 计算ncr
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pca(new pcl::PointCloud<pcl::PointXYZ>);
                for(const auto& idx : neighbor_indices){
                    const auto& neighbor = cloud_normal->points.at(idx);
                    pcl::PointXYZ point;
                    point.x = neighbor.x;
                    point.y = neighbor.y;
                    point.z = neighbor.z;
                    cloud_pca->points.push_back(point);
                }
                pca.setInputCloud(cloud_pca);

                Eigen::Vector3f eigValues = pca.getEigenValues();
                
        
                // 计算特征值和
                double sum = eigValues.sum();
                double eMin = eigValues.minCoeff();
                ncr_results[i] = eMin / sum ;
                // std::cout << " ncr : " <<ncr_results[i] << std::endl;
                if(ncr_results[i] < ncr_thresh_){
                        PointT pts;
                        pts.x = pt.x;
                        pts.y = pt.y;
                        pts.z = pt.z;
                        tmp_cloud_ncr.points.push_back(pts);
                }
            }
        }
        // std::cout << " ncr cal down " << std::endl;
    }
    
    clock_t t_aft_detect = clock();
    double t_detect = t_aft_detect  - t_bef_detect ;





    clock_t t_bef_assemble = clock();
    clock_t end = clock();
    time_taken_ = end - beg;

    if (params_.verbose)
    {
        const double sec = 1.0 / double(1000000);
        const double t_core = t_flush + t_czm + t_sort + t_pca + t_gle + t_revert + t_update;
        const double t_post = t_mrt + t_filter + t_convert + t_detect;
        const double t_accounted = t_core + t_post;
        cout << "[LidarProcess] total=" << time_taken_ * sec << "(sec)"
             << " input_pts=" << cloud_.points.size()
             << " nonground=" << cloud_nonground_.size()
             << " ground=" << cloud_ground_.size() << endl;
        cout << "  core: flush=" << t_flush * sec
             << " czm=" << t_czm * sec
             << " sort=" << t_sort * sec
             << " pca=" << t_pca * sec
             << " estimate=" << t_gle * sec
             << " revert=" << t_revert * sec
             << " update=" << t_update * sec
             << " subtotal=" << t_core * sec << endl;
        cout << "  post: mrt=" << t_mrt * sec
             << "(patches=" << mrt_patch_cnt << " no_nb=" << mrt_no_nb_cnt
             << " enable=" << params_.enable_MRT << ")"
             << " filter=" << t_filter * sec
             << "(enable=" << use_filter_ << ")"
             << " convert=" << t_convert * sec
             << "(enable=" << use_loc_trans_ << ")"
             << " detect=" << t_detect * sec
             << "(enable=" << use_lidar_detect_ << ")"
             << " subtotal=" << t_post * sec << endl;
        cout << "  accounted=" << t_accounted * sec
             << " unaccounted=" << (time_taken_ - t_accounted) * sec << endl;
    }

    // if (params_.verbose) cout << "\033[1;32m" << "PatchWorkpp::estimateGround() - Estimation is finished !" << "\033[0m" << endl;


    pcl::PointCloud<PointT> viz_cloud;
    pcl::PointCloud<PointT> tmp_cloud;
    // tmp_cloud.resize(cloud_nonground_.size() + cloud_ground_.size());
    // std::vector<int> foreground_idx;
    // int fusion_size = groud_cloud.points.size() + non_groud_cloud.points.size();
    for(int i=0;i<cloud_nonground_.size();i++){
        PointT point;
        point.x = cloud_nonground_[i].x;
        point.y = cloud_nonground_[i].y;
        point.z = cloud_nonground_[i].z;
        point.intensity = 255 ;
        if(point.z > filter_ground_noise_){ // 
            viz_cloud.push_back(point);
            tmp_cloud.push_back(point); // 通过非地面点方式传递给gridmap模块
        }
        PointT point_;
        point_.x = cloud_nonground_[i].x;
        point_.y = cloud_nonground_[i].y;
        point_.z = cloud_nonground_[i].z;
        point_.intensity = cloud_nonground_[i].intensity ;
        full_cloud.points.push_back(point_);
        
        foreground_idx.push_back(i);
    }
    // for(int i=0;i<cloud_nonground_new_.size();i++){
    //     PointT point;
    //     point.x = cloud_nonground_new_[i].x;
    //     point.y = cloud_nonground_new_[i].y;
    //     point.z = cloud_nonground_new_[i].z;
    //     point.intensity = 255 ;

    //     viz_cloud.push_back(point);
    //     PointT point_;
    //     point_.x = cloud_nonground_new_[i].x;
    //     point_.y = cloud_nonground_new_[i].y;
    //     point_.z = cloud_nonground_new_[i].z;
    //     point_.intensity = cloud_nonground_new_[i].intensity ;
    //     tmp_cloud.push_back(point_);
    //     foreground_idx.push_back(i);
    // }
    for(int i=0;i<cloud_ground_.size();i++){
        PointT point;
        point.x = cloud_ground_[i].x;
        point.y = cloud_ground_[i].y;
        point.z = cloud_ground_[i].z;
        point.intensity = 0 ;
        
        viz_cloud.push_back(point);
        PointT point_;
        point_.x = cloud_ground_[i].x;
        point_.y = cloud_ground_[i].y;
        point_.z = cloud_ground_[i].z;
        point_.intensity = cloud_ground_[i].intensity ;
        // tmp_cloud.points.push_back(point_);
        full_cloud.points.push_back(point_);
        
    }
    clock_t t_aft_assemble = clock();
    const double t_assemble = t_aft_assemble - t_bef_assemble;

    // pcl::PointCloud<pcl::PointXYZRGB> my_revert_viz_cloud;
    for(int i=0;i<my_revert_cloud.size();i++){
        pcl::PointXYZRGB point;
        point.x = my_revert_cloud[i].x;
        point.y = my_revert_cloud[i].y;
        point.z = my_revert_cloud[i].z;
        point.r = 255 ;
        point.g = 255 ;
        point.b = 0 ;

        patch_viz_cloud.push_back(point);        
    }


    // 下游 grid_map 订阅话题，不受 verbose 控制
    {
        sensor_msgs::msg::PointCloud2 cloud_ROS;
        pcl::toROSMsg(tmp_cloud, cloud_ROS);
        cloud_ROS.height = 1;
        cloud_ROS.width = tmp_cloud.points.size();
        cloud_ROS.header.stamp.sec = trunc(lidar_raw_data.time);
        cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
            (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
        cloud_ROS.header.frame_id = vis_frame_id;
        pub_ground_intensity->publish(cloud_ROS);
    }

    if(params_.verbose){
        clock_t t_bef_viz_pub = clock();
        sensor_msgs::msg::PointCloud2 cloud_ROS;
        pcl::toROSMsg(viz_cloud, cloud_ROS);
        cloud_ROS.header.stamp.sec =  trunc(lidar_raw_data.time);
        cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
        cloud_ROS.header.frame_id = vis_frame_id;
        pub_ground->publish(cloud_ROS);

        pcl::toROSMsg(tmp_cloud_ncr, cloud_ROS);
        cloud_ROS.height = 1;
        cloud_ROS.width = tmp_cloud_ncr.points.size();
        cloud_ROS.header.stamp.sec =  trunc(lidar_raw_data.time);
        cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
        cloud_ROS.header.frame_id = vis_frame_id;
        pub_ncr->publish(cloud_ROS);

        pcl::toROSMsg(convert_cloud, cloud_ROS);
        cloud_ROS.height = 1;
        cloud_ROS.width = convert_cloud.points.size();
        cloud_ROS.header.stamp.sec =  trunc(lidar_raw_data.time);
        cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
        cloud_ROS.header.frame_id = vis_frame_id;
        pub_ground_convert->publish(cloud_ROS);

        pcl::toROSMsg(patch_viz_cloud, cloud_ROS);
        cloud_ROS.height = 1;
        cloud_ROS.width = patch_viz_cloud.points.size();
        cloud_ROS.header.stamp.sec =  trunc(lidar_raw_data.time);
        cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
        cloud_ROS.header.frame_id = vis_frame_id;
        pub_patch->publish(cloud_ROS);
        


        if(cloud_revert.size() > 0){
            pcl::PointCloud<PointT> revert_cloud;
            for(int i=0;i<cloud_revert.size();i++){
                PointT point;
                point.x = cloud_revert[i].x;
                point.y = cloud_revert[i].y;
                point.z = cloud_revert[i].z;
                // point.intensity = 0 ;
                revert_cloud.points.push_back(point);                
            }
            pcl::toROSMsg(revert_cloud, cloud_ROS);
            cloud_ROS.header.stamp.sec =  trunc(lidar_raw_data.time);
            cloud_ROS.header.stamp.nanosec = static_cast<uint32_t>(
                                    (lidar_raw_data.time - trunc(lidar_raw_data.time)) * 1e9);
            cloud_ROS.header.frame_id = vis_frame_id;
            pub_ground_revert->publish(cloud_ROS);
        }

        pub_marker_->publish(mark_array);
        // pub_marker_->publish(mark_array_up);
        clock_t t_aft_viz_pub = clock();
        cout << "  viz: assemble=" << t_assemble / double(1000000)
             << "(sec) viz_pub=" << (t_aft_viz_pub - t_bef_viz_pub) / double(1000000)
             << "(sec) full_cloud=" << full_cloud.points.size() << endl;
    }



    return 0;
}

}
}