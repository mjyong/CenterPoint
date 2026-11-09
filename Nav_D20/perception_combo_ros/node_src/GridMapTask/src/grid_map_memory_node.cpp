#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <memory>
#include <queue>
#include <algorithm>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <mutex>
#include <limits>
#include <unordered_map>
#include <map>
#include <string>

#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "../include/logging.h"
#include "../include/grid_map_datapool.h"
#include <geometry_msgs/msg/pose_stamped.hpp>

class PerformanceTimer {
public:
    void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
    int64_t elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time_).count();
    }
    int64_t elapsedUs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

class GridMapTask : public rclcpp::Node
{
public:
    GridMapTask() : Node("grid_map_memory_node")
    {
        if (!loadParameters()) {
            LOG(ERROR) << "Failed to load parameters. Exiting.";
            throw std::runtime_error("Failed to load parameters.");
        }
        fusion_initialize();
        grid_initialize();
        octomap_initialize();
        LOG(INFO) << "Integrated node started successfully.";
    }

    ~GridMapTask() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - grid_node_start_time_);
        auto total_hours = std::chrono::duration_cast<std::chrono::hours>(total_duration);
        auto total_minutes = std::chrono::duration_cast<std::chrono::minutes>(total_duration % std::chrono::hours(1));
        auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(total_duration % std::chrono::minutes(1));
        LOG(INFO) << "=== Grid Module Statistics ===";
        LOG(INFO) << "Total run time: " << total_hours.count() << "h " 
                  << total_minutes.count() << "m " << total_seconds.count() << "s";
        LOG(INFO) << "Total callbacks processed: " << grid_total_callback_count_;
        if (grid_total_callback_count_ > 0) {
            LOG(INFO) << "Average callback time: " 
                      << (grid_total_callback_time_ / grid_total_callback_count_) << " ms";
            LOG(INFO) << "Max callback time: " << grid_max_callback_time_ << " ms";
            LOG(INFO) << "Min callback time: " << grid_min_callback_time_ << " ms";
        }
    }

    // 主循环：spin_some 收数后调用，按最新时间戳消费 depth/lidar 缓存
    void processPendingInputs();

private:
    // -------------------- Fusion 模块 --------------------
    std::string fusion_mode_;
    double fusion_time_sync_threshold_;
    int fusion_max_queue_size_;
    double fusion_rotation_x_, fusion_rotation_y_, fusion_rotation_z_;
    double fusion_translation_x_, fusion_translation_y_, fusion_translation_z_;
    std::string fusion_intermediate_frame_;
    std::string fusion_output_body_frame_;
    Eigen::Matrix4d fusion_intermediate_to_body_;
    Eigen::Matrix4d fusion_camera_to_body_;
    std::string fusion_depth_image_topic_;
    std::string fusion_camera_info_topic_;
    std::string fusion_lidar_topic_;
    double fusion_fx_, fusion_fy_, fusion_cx_, fusion_cy_;
    int fusion_depth_sample_step_;
    double fusion_depth_min_threshold_;
    double fusion_depth_max_threshold_;
    Eigen::Matrix4d fusion_transform_matrix_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr fusion_depth_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr fusion_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr fusion_lidar_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fusion_pointcloud_pub_;

    bool filter_enabled_;
    double filter_roi_x_min_, filter_roi_x_max_;
    double filter_roi_y_min_, filter_roi_y_max_;
    double filter_roi_z_min_, filter_roi_z_max_;
    double filter_radius_;
    int filter_min_neighbors_;
    std::string filter_raw_topic_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_raw_pub_;

    std::mutex fusion_data_mutex_;

    // 传感器订阅 QoS（yaml qos.*）；默认 BEST_EFFORT 以匹配板端点云/图像
    std::string sensor_qos_reliability_{"best_effort"};  // best_effort | reliable
    int sensor_qos_history_depth_{1};
    std::string sensor_qos_durability_{"volatile"};      // volatile | transient_local

    void fusion_initialize() {
        buildFusionTransformationMatrix();
        fusion_camera_to_body_ = fusion_intermediate_to_body_ * fusion_transform_matrix_;
        fusion_initializeByMode();
    }

    void buildFusionTransformationMatrix() {
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(fusion_rotation_z_, Eigen::Vector3d::UnitZ())
          * Eigen::AngleAxisd(fusion_rotation_y_, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(fusion_rotation_x_, Eigen::Vector3d::UnitX());
        fusion_transform_matrix_.setIdentity();
        fusion_transform_matrix_.block<3,3>(0,0) = R;
        fusion_transform_matrix_.block<3,1>(0,3) << fusion_translation_x_, fusion_translation_y_, fusion_translation_z_;
    }

    void fusion_initializeByMode() {
        fusion_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(grid_params_.fused_pointcloud_topic, 10);
        if (filter_enabled_) {
            filter_raw_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(filter_raw_topic_, 10);
            LOG(INFO) << "Filter enabled: ROI = [" << filter_roi_x_min_ << ", " << filter_roi_x_max_ << "] x ["
                      << filter_roi_y_min_ << ", " << filter_roi_y_max_ << "] x ["
                      << filter_roi_z_min_ << ", " << filter_roi_z_max_ << "], radius=" << filter_radius_
                      << ", min_neighbors=" << filter_min_neighbors_ << ", raw topic=" << filter_raw_topic_;
        }

        if (fusion_mode_ == "depth_only") {
            fusion_initializeDepthOnly();
            LOG(INFO) << "[Fusion Mode] depth_only";
        } else if (fusion_mode_ == "lidar_only") {
            fusion_initializeLidarOnly();
            LOG(INFO) << "[Fusion Mode] lidar_only";
        } else {
            fusion_initializeFusionMode();
            LOG(INFO) << "[Fusion Mode] fusion (depth + lidar)";
        }
    }

    static std::string toLowerAscii(std::string s) {
        for (char& c : s) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        return s;
    }

    void loadSensorQosFromYaml(const YAML::Node& config) {
        auto qos_node = config["qos"];
        if (!qos_node) return;
        sensor_qos_reliability_ = toLowerAscii(
            qos_node["reliability"].as<std::string>(sensor_qos_reliability_));
        sensor_qos_history_depth_ = qos_node["history_depth"].as<int>(sensor_qos_history_depth_);
        sensor_qos_durability_ = toLowerAscii(
            qos_node["durability"].as<std::string>(sensor_qos_durability_));
        if (sensor_qos_reliability_ != "best_effort" && sensor_qos_reliability_ != "reliable") {
            LOG(WARNING) << "[QoS] unknown reliability='" << sensor_qos_reliability_
                         << "', fallback to best_effort";
            sensor_qos_reliability_ = "best_effort";
        }
        if (sensor_qos_durability_ != "volatile" && sensor_qos_durability_ != "transient_local") {
            LOG(WARNING) << "[QoS] unknown durability='" << sensor_qos_durability_
                         << "', fallback to volatile";
            sensor_qos_durability_ = "volatile";
        }
        if (sensor_qos_history_depth_ < 1) {
            sensor_qos_history_depth_ = 1;
        }
    }

    // 须传 QoS 对象；传整型 depth 会默认 RELIABLE，与 BEST_EFFORT 发布端不匹配
    rclcpp::QoS buildSensorSubQos() const {
        rclcpp::QoS qos(rclcpp::KeepLast(static_cast<size_t>(sensor_qos_history_depth_)));
        if (sensor_qos_reliability_ == "reliable") {
            qos.reliable();
        } else {
            qos.best_effort();
        }
        if (sensor_qos_durability_ == "transient_local") {
            qos.transient_local();
        } else {
            qos.durability_volatile();
        }
        return qos;
    }

    void fusion_initializeDepthOnly() {
        const auto qos = buildSensorSubQos();
        fusion_depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            fusion_depth_image_topic_, qos,
            std::bind(&GridMapTask::fusion_depthImageCallback, this, std::placeholders::_1));
        fusion_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            fusion_camera_info_topic_, qos,
            std::bind(&GridMapTask::fusion_cameraInfoCallback, this, std::placeholders::_1));
        LOG(INFO) << "[QoS] depth/camera_info sub reliability=" << sensor_qos_reliability_
                  << " KeepLast(" << sensor_qos_history_depth_ << ") durability="
                  << sensor_qos_durability_;
    }

    void fusion_initializeLidarOnly() {
        const auto qos = buildSensorSubQos();
        fusion_lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            fusion_lidar_topic_, qos,
            std::bind(&GridMapTask::fusion_lidarCallback, this, std::placeholders::_1));
        LOG(INFO) << "[QoS] lidar sub reliability=" << sensor_qos_reliability_
                  << " KeepLast(" << sensor_qos_history_depth_ << ") durability="
                  << sensor_qos_durability_ << " on " << fusion_lidar_topic_;
    }

    void fusion_initializeFusionMode() {
        const auto qos = buildSensorSubQos();
        fusion_depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            fusion_depth_image_topic_, qos,
            std::bind(&GridMapTask::fusion_depthImageCallback, this, std::placeholders::_1));
        fusion_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            fusion_camera_info_topic_, qos,
            std::bind(&GridMapTask::fusion_cameraInfoCallback, this, std::placeholders::_1));
        fusion_lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            fusion_lidar_topic_, qos,
            std::bind(&GridMapTask::fusion_lidarCallback, this, std::placeholders::_1));
        LOG(INFO) << "[QoS] depth/lidar sub reliability=" << sensor_qos_reliability_
                  << " KeepLast(" << sensor_qos_history_depth_ << ") durability="
                  << sensor_qos_durability_;
    }

    void fusion_cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (msg->k.size() >= 4) {
            fusion_fx_ = msg->k[0];
            fusion_fy_ = msg->k[4];
            fusion_cx_ = msg->k[2];
            fusion_cy_ = msg->k[5];
        }
    }

    sensor_msgs::msg::PointCloud2::SharedPtr getFilteredPointCloud() {
        std::lock_guard<std::mutex> lock(fused_pc_mutex_);
        return latest_fused_pointcloud_;
    }

    void fusion_depthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (fusion_mode_ == "lidar_only") return;

        double depth_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double delay_ms = (system_now - depth_stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] Depth image received, stamp=" << depth_stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";

        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::FusionDepthData cached_newest;
        if (DP->main_data_.fusion_depth_cache.GetNewest(cached_newest) == 0 &&
            cached_newest.time >= depth_stamp) {
            return;
        }

        grid_map::FusionDepthData depth_data;
        depth_data.time = depth_stamp;
        depth_data.msg = msg;
        DP->main_data_.fusion_depth_cache.Push(depth_data);
    }

    void fusion_lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (fusion_mode_ == "depth_only") return;

        double lidar_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double delay_ms = (system_now - lidar_stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] LiDAR pointcloud received, stamp=" << lidar_stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";

        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::FusionLidarData cached_newest;
        if (DP->main_data_.fusion_lidar_cache.GetNewest(cached_newest) == 0 &&
            cached_newest.time >= lidar_stamp) {
            return;
        }

        grid_map::FusionLidarData lidar_data;
        lidar_data.time = lidar_stamp;
        lidar_data.msg = msg;
        DP->main_data_.fusion_lidar_cache.Push(lidar_data);
    }

    // lidar_only：body 变换 → 可选滤波 → 发布 fused 点云 → generateGrid
    void processLidarOnlyFrame(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
        auto transformed = fusion_applyBodyTransform(msg);
        updateLidarPointCloud(transformed);

        sensor_msgs::msg::PointCloud2::SharedPtr pc_to_publish;
        {
            std::lock_guard<std::mutex> lock(lidar_filtered_pc_mutex_);
            pc_to_publish = latest_lidar_pointcloud_filtered_;
        }
        if (!pc_to_publish) {
            pc_to_publish = transformed;
        }
        fusion_pointcloud_pub_->publish(*pc_to_publish);
        if (filter_enabled_ && filter_raw_pub_) {
            filter_raw_pub_->publish(*transformed);
        }
        {
            std::lock_guard<std::mutex> lock(fused_pc_mutex_);
            latest_fused_pointcloud_ = pc_to_publish;
        }
        auto now = this->now();
        auto dt = (now - pc_to_publish->header.stamp).seconds();
        LOG(INFO) << "[Output] Fused pointcloud (lidar_only) published, stamp="
                  << (pc_to_publish->header.stamp.sec + pc_to_publish->header.stamp.nanosec * 1e-9)
                  << ", system_time=" << now.seconds() << ", delay=" << dt * 1000.0 << " ms";
        // 用本帧滤波后雷达点云生成栅格（避免误用上一帧 fused）
        generateGrid(pc_to_publish->header, nullptr, pc_to_publish, nullptr);
    }

    void fusion_tryTimeSync() {
        std::lock_guard<std::mutex> lock(fusion_data_mutex_);
        grid_map::DataPool* DP = grid_map::DataPool::Instance();

        grid_map::FusionDepthData depth_data;
        grid_map::FusionLidarData lidar_data;
        if (DP->main_data_.fusion_depth_cache.GetNewest(depth_data) != 0 ||
            DP->main_data_.fusion_lidar_cache.GetNewest(lidar_data) != 0) {
            return;
        }

        double depth_time = depth_data.time;
        double lidar_time = lidar_data.time;
        double diff = std::abs(depth_time - lidar_time);

        if (diff < fusion_time_sync_threshold_) {
            LOG(INFO) << std::setprecision(13) << "[Fusion Sync] Depth stamp=" << depth_time
                      << ", LiDAR stamp=" << lidar_time << ", diff=" << diff << " s (synced)";
            processDepthForGrid(depth_data.msg, lidar_data.msg, nullptr);
            grid_map::FusionDepthData tmp_d;
            grid_map::FusionLidarData tmp_l;
            int d_discarded = 0, l_discarded = 0;
            DP->main_data_.fusion_depth_cache.PopNewest(tmp_d, &d_discarded);
            DP->main_data_.fusion_lidar_cache.PopNewest(tmp_l, &l_discarded);
            if (d_discarded > 0 || l_discarded > 0) {
                LOG(INFO) << "[Fusion Sync] discarded stale depth=" << d_discarded
                          << " lidar=" << l_discarded;
            }
        } else {
            LOG(INFO) << std::setprecision(13) << "[Fusion Sync] Depth stamp=" << depth_time
                      << ", LiDAR stamp=" << lidar_time << ", diff=" << diff << " s (NOT synced, processing LiDAR only)";
            fusion_processIndividualDataAndGenerateGrid();
        }
    }

    void fusion_processIndividualDataAndGenerateGrid() {
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::FusionLidarData lidar_data;
        if (DP->main_data_.fusion_lidar_cache.PopNewest(lidar_data) == 0) {
            auto transformed = fusion_applyBodyTransform(lidar_data.msg);
            updateLidarPointCloud(transformed);
            auto filtered_cloud = getFilteredPointCloud();
            if (!filtered_cloud) {
                LOG(WARNING) << "[individual LiDAR] No filtered pointcloud available, using raw transformed";
                filtered_cloud = transformed;
            }
            auto now = this->now();
            auto dt = (now - filtered_cloud->header.stamp).seconds();
            LOG(INFO) << "[Output] Fused pointcloud (individual LiDAR) published, stamp=" 
                      << (filtered_cloud->header.stamp.sec + filtered_cloud->header.stamp.nanosec*1e-9)
                      << ", system_time=" << now.seconds() << ", delay=" << dt*1000.0 << " ms";
            generateGrid(filtered_cloud->header, nullptr, filtered_cloud, nullptr);
        }
    }

    void processDepthForGrid(const sensor_msgs::msg::Image::SharedPtr depth_msg,
                             const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg,
                             const sensor_msgs::msg::PointCloud2::SharedPtr /*unused_fused*/)
    {
        PerformanceTimer total_timer;
        total_timer.start();
        
        double start_system_time = this->now().seconds();
        double depth_stamp = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
        LOG(INFO) << std::setprecision(13) << "[Fusion+Grid] processDepthForGrid start, depth_stamp=" << depth_stamp;

        PerformanceTimer fusion_timer;
        fusion_timer.start();
        sensor_msgs::msg::PointCloud2::SharedPtr final_fused;
        std::string fusion_source;
        if (lidar_msg) {
            auto depth_cloud = fusion_depthImageToPointCloudOptimized(depth_msg);
            final_fused = fusion_fusePointCloudsOptimized(depth_cloud, lidar_msg);
            fusion_source = "depth+lidar";
            LOG(INFO) << std::setprecision(13) << "[Fusion] Depth stamp: " << depth_stamp
                      << ", LiDAR stamp: " << (lidar_msg->header.stamp.sec + lidar_msg->header.stamp.nanosec*1e-9);
            auto lidar_transformed = fusion_applyBodyTransform(lidar_msg);
            updateLidarPointCloud(lidar_transformed);
        } else {
            final_fused = fusion_depthImageToPointCloudOptimized(depth_msg);
            fusion_source = "depth_only";
        }

        if (final_fused) {
            final_fused->header.frame_id = fusion_output_body_frame_;
            publishFilteredPointCloudAndUpdateGrid(final_fused);
        }
        double fusion_elapsed = fusion_timer.elapsedMs();
        LOG(INFO) << "[Timing] Fusion pointcloud generation (" << fusion_source << ") took " << fusion_elapsed << " ms";
        
        PerformanceTimer grid_timer;
        grid_timer.start();
        sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_for_grid = nullptr;
        if (lidar_msg) {
            pointcloud_for_grid = getFilteredPointCloud();
            if (!pointcloud_for_grid) {
                LOG(WARNING) << "[processDepthForGrid] Filtered pointcloud not available, using raw fused cloud";
                pointcloud_for_grid = final_fused;
            }
        }
        generateGrid(depth_msg->header, depth_msg, pointcloud_for_grid, nullptr);
        double grid_elapsed = grid_timer.elapsedMs();
        
        double total_elapsed = total_timer.elapsedMs();
        double end_system_time = this->now().seconds();
        double total_delay_ms = (end_system_time - depth_stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Timing] processDepthForGrid: fusion=" << fusion_elapsed
                  << " ms, grid=" << grid_elapsed << " ms, total=" << total_elapsed
                  << " ms, from depth_stamp to now=" << total_delay_ms << " ms";
        if (total_elapsed > 100.0) {
            LOG(WARNING) << "[Timing][SLOW] processDepthForGrid total=" << total_elapsed
                         << " ms exceeds 100 ms (fusion=" << fusion_elapsed
                         << ", grid(+memory)=" << grid_elapsed << ")";
        }
        grid_update_time_statistics(total_elapsed);
    }

    // 同步函数（以depth为基准）- 获取mask和RGB
    bool getSyncData(double depth_time, 
                     sensor_msgs::msg::Image::SharedPtr& mask_msg,
                     sensor_msgs::msg::CompressedImage::SharedPtr& rgb_msg) {
        grid_map::DataPool* DP = grid_map::DataPool::Instance();

        grid_map::MaskImageData mask_data;
        bool mask_found = false;
        double mask_diff_ms = std::numeric_limits<double>::max();
        if (DP->main_data_.mask_cache.GetClosestWithinThreshold(mask_data, depth_time, 
                grid_params_.time_sync_threshold * 1000) == 0) {
            mask_found = true;
            mask_diff_ms = std::abs(mask_data.time - depth_time) * 1000.0;
            mask_msg = mask_data.msg;
            LOG(INFO) << std::setprecision(13) << "[Sync] Mask stamp=" << mask_data.time
                      << ", depth_time=" << depth_time << ", diff=" << mask_diff_ms << " ms (within threshold)";
        } else {
            if (DP->main_data_.mask_cache.GetNewest(mask_data) == 0) {
                mask_diff_ms = std::abs(mask_data.time - depth_time) * 1000.0;
                mask_msg = mask_data.msg;
                mask_found = true;
                LOG(WARNING) << std::setprecision(13) << "[Sync] Mask stamp=" << mask_data.time
                             << ", depth_time=" << depth_time << ", diff=" << mask_diff_ms
                             << " ms (EXCEEDS threshold " << (grid_params_.time_sync_threshold * 1000) 
                             << " ms), using nearest available";
            } else {
                LOG(WARNING) << "[Sync] No mask message available in cache";
            }
        }

        grid_map::RgbImageData rgb_data;
        bool rgb_found = false;
        double rgb_diff_ms = std::numeric_limits<double>::max();
        if (DP->main_data_.rgb_image_cache.GetClosestWithinThreshold(rgb_data, depth_time,
                grid_params_.time_sync_threshold * 1000) == 0) {
            rgb_found = true;
            rgb_diff_ms = std::abs(rgb_data.time - depth_time) * 1000.0;
            rgb_msg = rgb_data.msg;
            LOG(INFO) << std::setprecision(13) << "[Sync] RGB stamp=" << rgb_data.time
                      << ", depth_time=" << depth_time << ", diff=" << rgb_diff_ms << " ms (within threshold)";
        } else {
            if (DP->main_data_.rgb_image_cache.GetNewest(rgb_data) == 0) {
                rgb_diff_ms = std::abs(rgb_data.time - depth_time) * 1000.0;
                rgb_msg = rgb_data.msg;
                rgb_found = true;
                LOG(WARNING) << std::setprecision(13) << "[Sync] RGB stamp=" << rgb_data.time
                             << ", depth_time=" << depth_time << ", diff=" << rgb_diff_ms
                             << " ms (EXCEEDS threshold), using nearest available";
            } else {
                LOG(INFO) << "[Sync] No RGB message available in cache";
            }
        }

        if (!mask_found) {
            LOG(WARNING) << "[Sync] No valid mask message for depth_time=" << depth_time;
            return false;
        }
        return true;
    }

    // 以RGB为基准的同步函数 - 获取mask和depth
    bool getSyncDataByRgb(double rgb_time,
                          sensor_msgs::msg::Image::SharedPtr& mask_msg,
                          sensor_msgs::msg::Image::SharedPtr& depth_msg,
                          double& total_error_ms) {
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        total_error_ms = std::numeric_limits<double>::max();

        // 1. 查找深度图
        grid_map::FusionDepthData depth_data;
        if (DP->main_data_.fusion_depth_cache.GetClosestWithinThreshold(depth_data, rgb_time,
                grid_params_.time_sync_threshold * 1000) != 0) {
            if (DP->main_data_.fusion_depth_cache.GetNewest(depth_data) != 0) {
                LOG(WARNING) << "[SyncByRgb] No depth message available in cache";
                return false;
            }
        }
        depth_msg = depth_data.msg;
        double depth_time = depth_data.time;
        double depth_diff_ms = std::abs(depth_time - rgb_time) * 1000.0;

        // 2. 查找mask
        grid_map::MaskImageData mask_data;
        if (DP->main_data_.mask_cache.GetClosestWithinThreshold(mask_data, rgb_time,
                grid_params_.time_sync_threshold * 1000) != 0) {
            if (DP->main_data_.mask_cache.GetNewest(mask_data) != 0) {
                LOG(WARNING) << "[SyncByRgb] No mask message available in cache";
                return false;
            }
        }
        mask_msg = mask_data.msg;
        double mask_time = mask_data.time;
        double mask_diff_ms = std::abs(mask_time - rgb_time) * 1000.0;

        total_error_ms = depth_diff_ms + mask_diff_ms;
        LOG(INFO) << std::setprecision(13) << "[SyncByRgb] RGB stamp=" << rgb_time
                  << ", depth stamp=" << depth_time << " (diff=" << depth_diff_ms << " ms)"
                  << ", mask stamp=" << mask_time << " (diff=" << mask_diff_ms << " ms)"
                  << ", total error=" << total_error_ms << " ms";
        return true;
    }

    // 以mask为基准的同步函数 - 获取depth和RGB
    bool getSyncDataByMask(double mask_time,
                           sensor_msgs::msg::Image::SharedPtr& depth_msg,
                           sensor_msgs::msg::CompressedImage::SharedPtr& rgb_msg,
                           double& total_error_ms) {
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        total_error_ms = std::numeric_limits<double>::max();

        // 1. 查找深度图
        grid_map::FusionDepthData depth_data;
        if (DP->main_data_.fusion_depth_cache.GetClosestWithinThreshold(depth_data, mask_time,
                grid_params_.time_sync_threshold * 1000) != 0) {
            if (DP->main_data_.fusion_depth_cache.GetNewest(depth_data) != 0) {
                LOG(WARNING) << "[SyncByMask] No depth message available in cache";
                return false;
            }
        }
        depth_msg = depth_data.msg;
        double depth_time = depth_data.time;
        double depth_diff_ms = std::abs(depth_time - mask_time) * 1000.0;

        // 2. 查找RGB图像
        grid_map::RgbImageData rgb_data;
        if (DP->main_data_.rgb_image_cache.GetClosestWithinThreshold(rgb_data, mask_time,
                grid_params_.time_sync_threshold * 1000) != 0) {
            if (DP->main_data_.rgb_image_cache.GetNewest(rgb_data) != 0) {
                LOG(WARNING) << "[SyncByMask] No RGB message available in cache";
                return false;
            }
        }
        rgb_msg = rgb_data.msg;
        double rgb_time = rgb_data.time;
        double rgb_diff_ms = std::abs(rgb_time - mask_time) * 1000.0;

        total_error_ms = depth_diff_ms + rgb_diff_ms;
        LOG(INFO) << std::setprecision(13) << "[SyncByMask] Mask stamp=" << mask_time
                  << ", depth stamp=" << depth_time << " (diff=" << depth_diff_ms << " ms)"
                  << ", RGB stamp=" << rgb_time << " (diff=" << rgb_diff_ms << " ms)"
                  << ", total error=" << total_error_ms << " ms";
        return true;
    }

    // 对指定类别的 label map 进行膨胀操作
    void dilateLabels(cv::Mat& label_map, int kernel_size, const std::vector<int>& target_labels) {
        if (kernel_size <= 0 || target_labels.empty()) return;
        cv::Mat original = label_map.clone();
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
        for (int label : target_labels) {
            cv::Mat mask = (original == label);
            if (cv::countNonZero(mask) == 0) continue;
            cv::Mat dilated_mask;
            cv::dilate(mask, dilated_mask, kernel);
            label_map.setTo(label, dilated_mask);
        }
    }

    // 统一的栅格生成接口（同步策略：优先以最新mask为基准）
    void generateGrid(const std_msgs::msg::Header& header,
                      const sensor_msgs::msg::Image::SharedPtr depth_msg,
                      const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg,
                      const sensor_msgs::msg::PointCloud2::SharedPtr /*reserved*/)
    {
        PerformanceTimer timer;
        timer.start();
        auto now_ms = [this]() { return getCurrentTimeMs(); };
        double t_sync = 0, t_rgb = 0, t_convert = 0, t_dilate = 0;
        double t_depth2grid = 0, t_proj_vis = 0, t_lidar_merge = 0, t_publish = 0;

        double stamp = header.stamp.sec + header.stamp.nanosec * 1e-9;
        LOG(INFO) << std::setprecision(13) << "[Grid] generateGrid using timestamp: " << stamp;

        sensor_msgs::msg::Image::SharedPtr selected_depth_msg = nullptr;
        sensor_msgs::msg::Image::SharedPtr selected_mask_msg = nullptr;
        sensor_msgs::msg::CompressedImage::SharedPtr selected_rgb_msg = nullptr;
        std::string sync_method = "none";
        double best_total_error = std::numeric_limits<double>::max();

        // ---- 1) 时间同步 ----
        double step = now_ms();
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::MaskImageData latest_mask_data;
        bool mask_based_success = false;
        if (DP->main_data_.mask_cache.GetNewest(latest_mask_data) == 0) {
            double mask_time = latest_mask_data.time;
            sensor_msgs::msg::Image::SharedPtr depth_msg0;
            sensor_msgs::msg::CompressedImage::SharedPtr rgb_msg0;
            double error0 = 0.0;
            if (getSyncDataByMask(mask_time, depth_msg0, rgb_msg0, error0)) {
                LOG(INFO) << "[Sync Strategy 0] Latest mask-based: total error = " << error0 << " ms";
                selected_depth_msg = depth_msg0;
                selected_mask_msg = latest_mask_data.msg;
                selected_rgb_msg = rgb_msg0;
                sync_method = "latest_mask_based";
                mask_based_success = true;
            } else {
                LOG(WARNING) << "[Sync Strategy 0] Failed to get depth or RGB for mask time " << mask_time;
            }
        }

        if (!mask_based_success) {
            if (depth_msg) {
                double depth_time = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
                sensor_msgs::msg::Image::SharedPtr mask_msg1;
                sensor_msgs::msg::CompressedImage::SharedPtr rgb_msg1;
                if (getSyncData(depth_time, mask_msg1, rgb_msg1)) {
                    double rgb_time1 = rgb_msg1 ? (rgb_msg1->header.stamp.sec + rgb_msg1->header.stamp.nanosec * 1e-9) : depth_time;
                    double mask_time1 = mask_msg1 ? (rclcpp::Time(mask_msg1->header.stamp).seconds()) : depth_time;
                    double error1 = std::abs(depth_time - rgb_time1) * 1000.0 + std::abs(depth_time - mask_time1) * 1000.0;
                    LOG(INFO) << "[Sync Strategy 1] Depth-based: total error = " << error1 << " ms";
                    if (error1 < best_total_error) {
                        best_total_error = error1;
                        selected_depth_msg = depth_msg;
                        selected_mask_msg = mask_msg1;
                        selected_rgb_msg = rgb_msg1;
                        sync_method = "depth_based";
                    }
                }
            }

            if (depth_msg) {
                double depth_time = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
                grid_map::RgbImageData rgb_data;
                if (DP->main_data_.rgb_image_cache.GetClosestWithinThreshold(rgb_data, depth_time,
                        grid_params_.time_sync_threshold * 1000) == 0 ||
                    DP->main_data_.rgb_image_cache.GetNewest(rgb_data) == 0) {
                    double rgb_time2 = rgb_data.time;
                    sensor_msgs::msg::Image::SharedPtr depth_msg2;
                    sensor_msgs::msg::Image::SharedPtr mask_msg2;
                    double error2 = 0.0;
                    if (getSyncDataByRgb(rgb_time2, mask_msg2, depth_msg2, error2)) {
                        LOG(INFO) << "[Sync Strategy 2] RGB-based (via depth time): total error = " << error2 << " ms";
                        if (error2 < best_total_error) {
                            best_total_error = error2;
                            selected_depth_msg = depth_msg2;
                            selected_mask_msg = mask_msg2;
                            selected_rgb_msg = rgb_data.msg;
                            sync_method = "rgb_based_via_depth";
                        }
                    }
                } else {
                    LOG(INFO) << "[Sync Strategy 2] No RGB message available in cache, skipping";
                }
            }
        }
        t_sync = now_ms() - step;

        bool has_mask = (selected_mask_msg != nullptr);
        if (!has_mask) {
            LOG(WARNING) << "[Grid] No valid mask message found via any sync method, falling back to pointcloud-only mode";
        }

        // ---- 2) RGB decode（仅 debug） ----
        step = now_ms();
        cv::Mat origin_image;
        if (grid_params_.publish_debug_vis) {
            if (selected_rgb_msg) {
                try {
                    cv_bridge::CvImagePtr rgb_bridge = cv_bridge::toCvCopy(selected_rgb_msg);
                    origin_image = rgb_bridge->image.clone();
                } catch (const cv_bridge::Exception& e) {
                    LOG(ERROR) << "cv_bridge exception (rgb compressed): " << e.what();
                }
            } else if (depth_msg) {
                grid_map::RgbImageData rgb_data;
                if (DP->main_data_.rgb_image_cache.GetNewest(rgb_data) == 0) {
                    try {
                        cv_bridge::CvImagePtr rgb_bridge = cv_bridge::toCvCopy(rgb_data.msg);
                        origin_image = rgb_bridge->image.clone();
                        LOG(INFO) << "[Grid] Using newest RGB image for visualization (no sync)";
                    } catch (const cv_bridge::Exception& e) {
                        LOG(ERROR) << "cv_bridge exception (rgb compressed fallback): " << e.what();
                    }
                }
            }
        }
        t_rgb = now_ms() - step;

        std::string grid_source;

        // ---------- 有分割mask ----------
        if (has_mask && selected_depth_msg) {
            grid_source = "mask+depth";
            LOG(INFO) << "[Grid Mode] Using mask + depth (sync method: " << sync_method << ")";

            // ---- 3) depth/mask convert ----
            step = now_ms();
            cv_bridge::CvImagePtr depth_bridge = cv_bridge::toCvCopy(selected_depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv::Mat mask = cv_bridge::toCvCopy(selected_mask_msg, sensor_msgs::image_encodings::MONO8)->image;
            cv::Mat label_map;
            mask.convertTo(label_map, CV_32SC1);
            t_convert = now_ms() - step;

            // ---- 4) mask dilate ----
            step = now_ms();
            if (grid_params_.freespace_boundary_line_width > 0 && !grid_params_.expand_class_labels.empty()) {
                int kernel_size = 2 * grid_params_.freespace_boundary_line_width + 1;
                dilateLabels(label_map, kernel_size, grid_params_.expand_class_labels);
                LOG(INFO) << "[Mask Dilate] dilated classes: " << grid_params_.expand_class_labels.size()
                          << ", kernel size=" << kernel_size;
            }
            t_dilate = now_ms() - step;

            std::vector<cv::Point2f> proj_points;
            std::vector<int> point_labels;

            // ---- 5) DepthToGrid ----
            step = now_ms();
            auto [depth_cloud, ground_depth, non_ground_depth, grid_depth] =
                grid_generate_point_cloud_and_grid(depth_bridge->image, mask, label_map, header, false, proj_points, point_labels);
            t_depth2grid = now_ms() - step;

            // ---- 6) projected points vis ----
            step = now_ms();
            if (grid_params_.publish_debug_vis && !origin_image.empty() && !proj_points.empty()) {
                cv::Mat vis_image = origin_image.clone();
                drawProjectedPoints(vis_image, proj_points, point_labels, grid_params_.freespace_classes);
                publishProjectedPointsImage(vis_image, selected_depth_msg->header);
            }
            t_proj_vis = now_ms() - step;

            // ---- 7) optional lidar grid merge ----
            step = now_ms();
            if (grid_params_.enable_fusion_with_pointcloud) {
                sensor_msgs::msg::PointCloud2::SharedPtr lidar_pc_to_use = nullptr;
                {
                    std::lock_guard<std::mutex> lock(lidar_filtered_pc_mutex_);
                    if (latest_lidar_pointcloud_filtered_) {
                        lidar_pc_to_use = latest_lidar_pointcloud_filtered_;
                    } else {
                        std::lock_guard<std::mutex> lock2(lidar_pc_mutex_);
                        lidar_pc_to_use = latest_lidar_pointcloud_;
                    }
                }
                if (lidar_pc_to_use) {
                    grid_source = "mask+depth+lidar_only";
                    LOG(INFO) << "[Grid Mode] mask + depth + pure LiDAR pointcloud";
                    nav_msgs::msg::OccupancyGrid grid_pc;
                    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pc(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_pc(new pcl::PointCloud<pcl::PointXYZ>);
                    grid_generate_from_pointcloud(lidar_pc_to_use, header, grid_pc, ground_pc, non_ground_pc);

                    for (size_t i = 0; i < grid_depth.data.size(); ++i) {
                        if (grid_pc.data[i] > grid_depth.data[i]) grid_depth.data[i] = grid_pc.data[i];
                    }
                    *ground_depth += *ground_pc;
                    *non_ground_depth += *non_ground_pc;
                    LOG(INFO) << "[Grid] Merged pure LiDAR grid with depth grid";
                }
            }
            t_lidar_merge = now_ms() - step;

            // ---- 8) publish (+ memory) ----
            step = now_ms();
            pcl::PointCloud<pcl::PointXYZ>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            *total_cloud = *ground_depth + *non_ground_depth;
            grid_publish_results(total_cloud, ground_depth, non_ground_depth, grid_depth, origin_image, header, grid_source);
            t_publish = now_ms() - step;
        }
        // ---------- 无分割结果 ----------
        else {
            sensor_msgs::msg::PointCloud2::SharedPtr pc_to_use = pointcloud_msg;
            if (!pc_to_use) {
                pc_to_use = getFilteredPointCloud();
            }
            if (!pc_to_use) {
                std::lock_guard<std::mutex> lock(lidar_pc_mutex_);
                if (latest_lidar_pointcloud_) pc_to_use = latest_lidar_pointcloud_;
            }

            if (pc_to_use) {
                grid_source = "pointcloud_only";
                LOG(INFO) << "[Grid Mode] pointcloud only (no mask)";
                step = now_ms();
                nav_msgs::msg::OccupancyGrid grid_pc;
                pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pc(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_pc(new pcl::PointCloud<pcl::PointXYZ>);
                grid_generate_from_pointcloud(pc_to_use, header, grid_pc, ground_pc, non_ground_pc);
                t_depth2grid = now_ms() - step;  // reuse field for pc→grid

                step = now_ms();
                pcl::PointCloud<pcl::PointXYZ>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                *total_cloud = *ground_pc + *non_ground_pc;
                grid_publish_results(total_cloud, ground_pc, non_ground_pc, grid_pc, origin_image, header, grid_source);
                t_publish = now_ms() - step;
            } else {
                LOG(WARNING) << "[Grid] No mask and no pointcloud available, skipping grid generation";
                grid_update_time_statistics(timer.elapsedMs());
                return;
            }
        }

        double elapsed = timer.elapsedMs();
        double wall_sum = t_sync + t_rgb + t_convert + t_dilate + t_depth2grid + t_proj_vis + t_lidar_merge + t_publish;
        LOG(INFO) << "[Timing] generateGrid (" << grid_source << ") took " << elapsed << " ms"
                  << " | sync=" << t_sync
                  << " rgb_decode=" << t_rgb
                  << " convert=" << t_convert
                  << " dilate=" << t_dilate
                  << " depth2grid=" << t_depth2grid
                  << " proj_vis=" << t_proj_vis
                  << " lidar_merge=" << t_lidar_merge
                  << " publish(+mem)=" << t_publish
                  << " sum=" << wall_sum
                  << " (source=" << sync_method << ")";
        if (elapsed > 100.0) {
            LOG(WARNING) << "[Timing][SLOW] generateGrid (" << grid_source << ") took=" << elapsed
                         << " ms exceeds 100 ms"
                         << " | sync=" << t_sync
                         << " convert=" << t_convert
                         << " dilate=" << t_dilate
                         << " depth2grid=" << t_depth2grid
                         << " proj_vis=" << t_proj_vis
                         << " lidar_merge=" << t_lidar_merge
                         << " publish(+mem)=" << t_publish;
        }
        grid_update_time_statistics(elapsed);
    }

    sensor_msgs::msg::PointCloud2::SharedPtr fusion_depthImageToPointCloudOptimized(
        const sensor_msgs::msg::Image::SharedPtr header_msg)
    {
        PerformanceTimer timer;
        timer.start();
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(header_msg, header_msg->encoding);
        const cv::Mat& depth_image = cv_ptr->image;
        const int rows = depth_image.rows;
        const int cols = depth_image.cols;
        const int step = fusion_depth_sample_step_;
        const int estimated_points = ((rows + step - 1) / step) * ((cols + step - 1) / step);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl_cloud->reserve(estimated_points);
        
        const bool is_uint16 = (header_msg->encoding == "16UC1" || header_msg->encoding == "mono16");
        const bool is_uint8  = (header_msg->encoding == "mono8" || header_msg->encoding == "8UC1");
        const bool is_float32 = (header_msg->encoding == "32FC1");

        if (!is_uint16 && !is_uint8 && !is_float32) {
            LOG(ERROR) << "Unsupported depth encoding: " << header_msg->encoding;
            return std::make_shared<sensor_msgs::msg::PointCloud2>();
        }

        float depth_scale = 1.0f;
        if (header_msg->encoding == "16UC1") {
            depth_scale = 0.001f;
        } else if (header_msg->encoding == "mono16") {
            depth_scale = 0.000125f;
        } else if (is_uint8) {
            depth_scale = 0.01;
        } else if (is_float32) {
            depth_scale = 1.0f;
        }

        const Eigen::Matrix4d& T = fusion_camera_to_body_;
        const double fx = fusion_fx_, fy = fusion_fy_, cx = fusion_cx_, cy = fusion_cy_;
        const double min_depth = fusion_depth_min_threshold_;
        const double max_depth = fusion_depth_max_threshold_;
        
        if (is_uint16) {
            const uint16_t* depth_data = depth_image.ptr<uint16_t>();
            for (int v = 0; v < rows; v += step) {
                const int row_offset = v * cols;
                for (int u = 0; u < cols; u += step) {
                    uint16_t d16 = depth_data[row_offset + u];
                    if (d16 == 0) continue;
                    float depth = d16 * depth_scale;
                    if (depth <= min_depth || depth > max_depth) continue;
                    double x_cam = (u - cx) * depth / fx;
                    double y_cam = (v - cy) * depth / fy;
                    double z_cam = depth;
                    Eigen::Vector4d p_cam(x_cam, y_cam, z_cam, 1.0);
                    Eigen::Vector4d p_body = T * p_cam;
                    pcl::PointXYZI point;
                    point.x = p_body(0);
                    point.y = p_body(1);
                    point.z = p_body(2);
                    point.intensity = depth;
                    pcl_cloud->points.push_back(point);
                }
            }
        } 
        else if (is_uint8) {
            const uint8_t* depth_data = depth_image.ptr<uint8_t>();
            for (int v = 0; v < rows; v += step) {
                const int row_offset = v * cols;
                for (int u = 0; u < cols; u += step) {
                    uint8_t d8 = depth_data[row_offset + u];
                    if (d8 == 0) continue;
                    float depth = static_cast<float>(d8) * depth_scale;
                    if (depth <= min_depth || depth > max_depth) continue;
                    double x_cam = (u - cx) * depth / fx;
                    double y_cam = (v - cy) * depth / fy;
                    double z_cam = depth;
                    Eigen::Vector4d p_cam(x_cam, y_cam, z_cam, 1.0);
                    Eigen::Vector4d p_body = T * p_cam;
                    pcl::PointXYZI point;
                    point.x = p_body(0);
                    point.y = p_body(1);
                    point.z = p_body(2);
                    point.intensity = depth;
                    pcl_cloud->points.push_back(point);
                }
            }
        }
        else {
            const float* depth_data = depth_image.ptr<float>();
            for (int v = 0; v < rows; v += step) {
                const int row_offset = v * cols;
                for (int u = 0; u < cols; u += step) {
                    float depth = depth_data[row_offset + u];
                    if (std::isnan(depth) || depth <= min_depth || depth > max_depth) continue;
                    double x_cam = (u - cx) * depth / fx;
                    double y_cam = (v - cy) * depth / fy;
                    double z_cam = depth;
                    Eigen::Vector4d p_cam(x_cam, y_cam, z_cam, 1.0);
                    Eigen::Vector4d p_body = T * p_cam;
                    pcl::PointXYZI point;
                    point.x = p_body(0);
                    point.y = p_body(1);
                    point.z = p_body(2);
                    point.intensity = depth;
                    pcl_cloud->points.push_back(point);
                }
            }
        }
        
        auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*pcl_cloud, *cloud_msg);
        cloud_msg->header = header_msg->header;
        cloud_msg->header.frame_id = fusion_output_body_frame_;
        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[Depth2PointCloud] Generated " << pcl_cloud->size() << " points, took " << elapsed << " ms";
        return cloud_msg;
    }
    
    sensor_msgs::msg::PointCloud2::SharedPtr fusion_fusePointCloudsOptimized(
        const sensor_msgs::msg::PointCloud2::SharedPtr depth_cloud_body,
        const sensor_msgs::msg::PointCloud2::SharedPtr lidar_cloud)
    {
        PerformanceTimer timer;
        timer.start();
        pcl::PointCloud<pcl::PointXYZI> lidar_pcl;
        pcl::fromROSMsg(*lidar_cloud, lidar_pcl);
        pcl::PointCloud<pcl::PointXYZI> lidar_body;
        lidar_body.reserve(lidar_pcl.size());
        Eigen::Matrix4f T_body = fusion_intermediate_to_body_.cast<float>();
        pcl::transformPointCloud(lidar_pcl, lidar_body, T_body);
        
        pcl::PointCloud<pcl::PointXYZI> depth_pcl;
        pcl::fromROSMsg(*depth_cloud_body, depth_pcl);
        
        pcl::PointCloud<pcl::PointXYZI> fused;
        fused.reserve(lidar_body.size() + depth_pcl.size());
        fused.insert(fused.end(), lidar_body.begin(), lidar_body.end());
        fused.insert(fused.end(), depth_pcl.begin(), depth_pcl.end());
        
        auto fused_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(fused, *fused_msg);
        fused_msg->header = depth_cloud_body->header;
        fused_msg->header.frame_id = fusion_output_body_frame_;
        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[Fusion] Fused pointcloud: depth=" << depth_pcl.size() << ", lidar=" << lidar_body.size()
                  << ", total=" << fused.size() << ", took " << elapsed << " ms";
        return fused_msg;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr fusion_applyBodyTransform(
        const sensor_msgs::msg::PointCloud2::SharedPtr cloud)
    {
        if (fusion_intermediate_to_body_.isIdentity()) {
            auto out = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
            out->header.frame_id = fusion_output_body_frame_;
            return out;
        }
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(*cloud, pcl_cloud);
        pcl::transformPointCloud(pcl_cloud, pcl_cloud, fusion_intermediate_to_body_);
        auto out = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(pcl_cloud, *out);
        out->header = cloud->header;
        out->header.frame_id = fusion_output_body_frame_;
        return out;
    }

    void publishFilteredPointCloudAndUpdateGrid(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
        PerformanceTimer timer;
        timer.start();
        sensor_msgs::msg::PointCloud2::SharedPtr to_publish;
        if (!filter_enabled_) {
            to_publish = cloud;
            fusion_pointcloud_pub_->publish(*to_publish);
            LOG(INFO) << "[Filter] Filter disabled, publishing raw pointcloud";
        } else {
            filter_raw_pub_->publish(*cloud);
            auto filtered = applyDensityFilterInROI(cloud);
            fusion_pointcloud_pub_->publish(*filtered);
            to_publish = filtered;
            LOG(INFO) << "[Filter] Filter enabled, published filtered pointcloud";
        }
        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[Timing] publishFilteredPointCloudAndUpdateGrid took " << elapsed << " ms";
        {
            std::lock_guard<std::mutex> lock(fused_pc_mutex_);
            latest_fused_pointcloud_ = to_publish;
            LOG(INFO) << std::setprecision(13) << "[Fusion] Updated latest_fused_pointcloud_ with timestamp: "
                      << to_publish->header.stamp.sec + to_publish->header.stamp.nanosec*1e-9;
        }
    }

    sensor_msgs::msg::PointCloud2::SharedPtr applyDensityFilterInROI(
        const sensor_msgs::msg::PointCloud2::SharedPtr cloud)
    {
        PerformanceTimer timer;
        timer.start();
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud, *pcl_cloud);
        size_t original_size = pcl_cloud->size();
        if (pcl_cloud->empty()) {
            LOG(INFO) << "[Filter] Input pointcloud is empty, skipping filter";
            return cloud;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr ground_region_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr roi_non_ground_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr outside_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        ground_region_cloud->reserve(pcl_cloud->size());
        roi_non_ground_cloud->reserve(pcl_cloud->size());
        outside_cloud->reserve(pcl_cloud->size());

        const double gx_min = grid_params_.ground_region_x_min_;
        const double gx_max = grid_params_.ground_region_x_max_;
        const double gy_min = grid_params_.ground_region_y_min_;
        const double gy_max = grid_params_.ground_region_y_max_;

        for (const auto& pt : pcl_cloud->points) {
            bool in_ground_region = (pt.x >= gx_min && pt.x <= gx_max &&
                                     pt.y >= gy_min && pt.y <= gy_max);
            if (in_ground_region) {
                ground_region_cloud->push_back(pt);
                continue;
            }

            bool in_roi = (pt.x >= filter_roi_x_min_ && pt.x <= filter_roi_x_max_ &&
                           pt.y >= filter_roi_y_min_ && pt.y <= filter_roi_y_max_ &&
                           pt.z >= filter_roi_z_min_ && pt.z <= filter_roi_z_max_);
            if (in_roi) {
                roi_non_ground_cloud->push_back(pt);
            } else {
                outside_cloud->push_back(pt);
            }
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_roi(new pcl::PointCloud<pcl::PointXYZI>);
        size_t roi_before = roi_non_ground_cloud->size();
        if (!roi_non_ground_cloud->empty()) {
            pcl::RadiusOutlierRemoval<pcl::PointXYZI> radius_filter;
            radius_filter.setInputCloud(roi_non_ground_cloud);
            radius_filter.setRadiusSearch(filter_radius_);
            radius_filter.setMinNeighborsInRadius(filter_min_neighbors_);
            radius_filter.setKeepOrganized(false);
            radius_filter.filter(*filtered_roi);
            LOG(INFO) << "[Filter] ROI filter: ROI range x=[" << filter_roi_x_min_ << "," << filter_roi_x_max_
                      << "], y=[" << filter_roi_y_min_ << "," << filter_roi_y_max_
                      << "], z=[" << filter_roi_z_min_ << "," << filter_roi_z_max_
                      << "], radius=" << filter_radius_ << ", min_neighbors=" << filter_min_neighbors_
                      << ", points_before=" << roi_before << ", points_after=" << filtered_roi->size()
                      << ", removed=" << (roi_before - filtered_roi->size());
        } else {
            filtered_roi = roi_non_ground_cloud;
            LOG(INFO) << "[Filter] No points in ROI, skipping filter";
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        result_cloud->reserve(ground_region_cloud->size() + filtered_roi->size() + outside_cloud->size());
        *result_cloud = *ground_region_cloud;
        *result_cloud += *filtered_roi;
        *result_cloud += *outside_cloud;

        auto out_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*result_cloud, *out_msg);
        out_msg->header = cloud->header;
        
        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[Filter] Total points: original=" << original_size << ", after_filter=" << result_cloud->size()
                  << ", took " << elapsed << " ms";
        return out_msg;
    }

    void updateLidarPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_raw_body) {
        {
            std::lock_guard<std::mutex> lock(lidar_pc_mutex_);
            latest_lidar_pointcloud_ = cloud_raw_body;
        }

        if (filter_enabled_) {
            auto filtered = applyDensityFilterInROI(cloud_raw_body);
            std::lock_guard<std::mutex> lock(lidar_filtered_pc_mutex_);
            latest_lidar_pointcloud_filtered_ = filtered;
        } else {
            std::lock_guard<std::mutex> lock(lidar_filtered_pc_mutex_);
            latest_lidar_pointcloud_filtered_ = cloud_raw_body;
        }
    }

    // -------------------- Grid 模块 --------------------
    struct GridParameters {
        std::string rgb_topic;
        std::string depth_topic;
        std::string mask_topic;                     // 新的mask话题
        std::string pointcloud_topic;
        std::string occupancy_grid_topic;
        std::string visualization_topic;
        std::string ground_pointcloud_topic;
        std::string non_ground_pointcloud_topic;
        std::string fused_pointcloud_topic;
        std::string rgb_image_type;
        std::string depth_image_type;
        double fx, fy, cx, cy;
        double yaw, pitch, roll, pitch2;
        double x_offset, y_offset, z_offset;
        int max_queue_size;
        double time_sync_threshold;
        double depth_scale;
        double depth_min, depth_max;
        int skip_pixels;
        double point_cloud_range_min_x, point_cloud_range_max_x;
        double point_cloud_range_min_y, point_cloud_range_max_y;
        double point_cloud_range_min_z;
        double point_cloud_range_max_z;
        double freespace_max_z, freespace_min_z, freespace_mini_z;
        std::vector<std::string> freespace_classes;
        int expand_pixels;
        int freespace_boundary_line_width;          // 膨胀核半径（像素）
        std::vector<std::string> expand_classes;    // 需要膨胀的类别名称列表
        double ground_region_x_min_, ground_region_x_max_;
        double ground_region_y_min_, ground_region_y_max_;
        float grid_resolution;
        int grid_size;
        bool enable_fusion_with_pointcloud;
        bool lidar_only_mode;
        std::string projected_points_vis_topic;
        // 调试可视化（RGB decode / 投影图 / seg vis）；板端建议 false
        bool publish_debug_vis = false;
        // 发布 total/ground/non_ground 点云（OctoMap 单帧仍依赖后两者）
        bool publish_split_clouds = false;
        // 辅助：存储需要膨胀的类别对应的标签值（int）
        std::vector<int> expand_class_labels;
    } grid_params_;

    Eigen::Matrix4f grid_transformation_matrix_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr grid_rgb_raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr grid_rgb_compressed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr grid_mask_sub_;   // mask图像订阅

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grid_point_cloud_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_occupancy_grid_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr grid_visualization_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grid_ground_point_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grid_non_ground_point_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr projected_points_pub_;

    std::chrono::high_resolution_clock::time_point grid_node_start_time_;
    std::chrono::high_resolution_clock::time_point grid_last_statistics_time_;
    size_t grid_total_callback_count_ = 0;
    double grid_total_callback_time_ = 0.0;
    double grid_max_callback_time_ = 0.0;
    double grid_min_callback_time_ = std::numeric_limits<double>::max();
    size_t grid_statistics_interval_ = 100;
    std::mutex grid_mutex_;

    std::mutex fused_pc_mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_fused_pointcloud_;
    std::mutex lidar_pc_mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_lidar_pointcloud_;
    std::mutex lidar_filtered_pc_mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_lidar_pointcloud_filtered_;

    // 新增：类别映射和可通行类别列表
    std::map<int, std::string> class_map_;
    std::vector<std::string> freespace_classes_list_;

    void grid_initialize() {
        grid_initialize_transformation_matrix();
        grid_create_subscribers_publishers();
        grid_node_start_time_ = std::chrono::high_resolution_clock::now();
        grid_last_statistics_time_ = grid_node_start_time_;
    }

    void grid_initialize_transformation_matrix() {
        Eigen::Matrix3f R_yaw, R_pitch, R_roll, R_pitch2;
        R_yaw << cos(grid_params_.yaw), -sin(grid_params_.yaw), 0,
                 sin(grid_params_.yaw),  cos(grid_params_.yaw), 0,
                 0, 0, 1;
        R_pitch << cos(grid_params_.pitch), 0, sin(grid_params_.pitch),
                   0, 1, 0,
                   -sin(grid_params_.pitch), 0, cos(grid_params_.pitch);
        R_pitch2 << cos(grid_params_.pitch2), 0, sin(grid_params_.pitch2),
                    0, 1, 0,
                    -sin(grid_params_.pitch2), 0, cos(grid_params_.pitch2);
        R_roll << 1, 0, 0,
                 0, cos(grid_params_.roll), -sin(grid_params_.roll),
                 0, sin(grid_params_.roll),  cos(grid_params_.roll);
        grid_transformation_matrix_.block<3,3>(0,0) = R_pitch2 * R_roll * R_pitch;
        grid_transformation_matrix_.block<3,1>(0,3) << grid_params_.x_offset, grid_params_.y_offset, grid_params_.z_offset;
    }

    void grid_create_subscribers_publishers() {
        const auto qos = buildSensorSubQos();

        if (grid_params_.rgb_image_type == "raw") {
            grid_rgb_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                grid_params_.rgb_topic, qos,
                std::bind(&GridMapTask::grid_rgb_raw_callback, this, std::placeholders::_1));
            LOG(INFO) << "Subscribing to raw RGB image on " << grid_params_.rgb_topic;
        } else {
            grid_rgb_compressed_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                grid_params_.rgb_topic, qos,
                std::bind(&GridMapTask::grid_rgb_compressed_callback, this, std::placeholders::_1));
            LOG(INFO) << "Subscribing to compressed RGB image on " << grid_params_.rgb_topic;
        }

        // 订阅mask图像（单通道，MONO8编码，0=可通行，1..N=障碍物类别，255=背景）
        grid_mask_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            grid_params_.mask_topic, qos,
            std::bind(&GridMapTask::grid_mask_callback, this, std::placeholders::_1));
        LOG(INFO) << "Subscribing to mask image on " << grid_params_.mask_topic;

        grid_point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            grid_params_.pointcloud_topic, 10);
        grid_occupancy_grid_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            grid_params_.occupancy_grid_topic, 10);
        grid_visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            grid_params_.visualization_topic, 10);
        grid_ground_point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            grid_params_.ground_pointcloud_topic, 10);
        grid_non_ground_point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            grid_params_.non_ground_pointcloud_topic, 10);
        
        projected_points_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            grid_params_.projected_points_vis_topic, 10);
        LOG(INFO) << "Publishing projected points visualization on " << grid_params_.projected_points_vis_topic;
    }

    void grid_rgb_compressed_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double delay_ms = (system_now - stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] RGB compressed received, stamp=" << stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";
        
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::RgbImageData rgb_data;
        rgb_data.time = stamp;
        rgb_data.msg = msg;
        DP->main_data_.rgb_image_cache.Push(rgb_data);
    }

    void grid_rgb_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double delay_ms = (system_now - stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] RGB raw received, stamp=" << stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";
        
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& e) {
            LOG(ERROR) << "cv_bridge exception (rgb raw): " << e.what();
            return;
        }
        sensor_msgs::msg::CompressedImage compressed_msg;
        compressed_msg.header = msg->header;
        compressed_msg.format = "jpeg";
        std::vector<uchar> buffer;
        cv::imencode(".jpg", cv_ptr->image, buffer);
        compressed_msg.data = buffer;
        auto compressed_ptr = std::make_shared<sensor_msgs::msg::CompressedImage>(compressed_msg);

        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::RgbImageData rgb_data;
        rgb_data.time = stamp;
        rgb_data.msg = compressed_ptr;
        DP->main_data_.rgb_image_cache.Push(rgb_data);
    }

    // 新的mask回调函数
    void grid_mask_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double delay_ms = (system_now - stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] Mask image received, stamp=" << stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";
        
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::MaskImageData mask_data;
        mask_data.time = stamp;
        mask_data.msg = msg;
        DP->main_data_.mask_cache.Push(mask_data);
    }

    void grid_publish_results(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
                              const nav_msgs::msg::OccupancyGrid& grid,
                              const cv::Mat& vis_image,
                              const std_msgs::msg::Header& header,
                              const std::string& source)
    {
        double grid_stamp = grid.header.stamp.sec + grid.header.stamp.nanosec * 1e-9;
        double system_now = this->now().seconds();
        double grid_delay = (system_now - grid_stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Output] Occupancy grid published, stamp=" << grid_stamp
                  << ", system_time=" << system_now << ", delay=" << grid_delay << " ms, source=" << source;
        
        if (octomap_params_.enable_accumulation) {
            grid_map::DataPool* DP = grid_map::DataPool::Instance();
            grid_map::OdomData odom_data;
            double frame_time = header.stamp.sec + header.stamp.nanosec * 1e-9;
            double tol_ms = octomap_params_.time_tolerance * 1000.0;
            if (DP->main_data_.odom_cache.GetClosestWithinThreshold(odom_data, frame_time, tol_ms) == 0) {
                PerformanceTimer mem_timer;
                mem_timer.start();
                double sys_now = this->now().seconds();
                memoryCheckPeriodicClear(sys_now);

                size_t mem_ins = memoryInsertFromCloud(non_ground, odom_data.pose);
                size_t mem_prn = memoryPruneOutside(odom_data.pose);
                nav_msgs::msg::OccupancyGrid merged_grid = grid;
                memoryMergeToBody(merged_grid, odom_data.pose);
                grid_occupancy_grid_publisher_->publish(merged_grid);
                double mem_ms = mem_timer.elapsedMs();
                if (octomap_params_.enable_profiling) {
                    LOG(INFO) << "[Memory] insert=" << mem_ins << " prune=" << mem_prn
                              << " voxels=" << memory_voxel_map_.size() << " took=" << mem_ms << " ms";
                }
                if (mem_ms > 100.0) {
                    LOG(WARNING) << "[Timing][SLOW] Memory took=" << mem_ms
                                 << " ms exceeds 100 ms (insert=" << mem_ins
                                 << ", voxels=" << memory_voxel_map_.size() << ")";
                }
            } else {
                grid_occupancy_grid_publisher_->publish(grid);
                LOG(WARNING) << "[Memory] No odom for memory merge, publishing raw grid";
            }
        } else {
            grid_occupancy_grid_publisher_->publish(grid);
        }

        if (grid_params_.publish_debug_vis && !vis_image.empty()) {
            sensor_msgs::msg::Image ros_img;
            cv_bridge::CvImage(header, "bgr8", vis_image).toImageMsg(ros_img);
            ros_img.header.frame_id = "camera";
            grid_visualization_pub_->publish(ros_img);
            double vis_stamp = ros_img.header.stamp.sec + ros_img.header.stamp.nanosec * 1e-9;
            double vis_delay = (this->now().seconds() - vis_stamp) * 1000.0;
            LOG(INFO) << "[Output] Visualization image published, stamp=" << vis_stamp
                      << ", delay=" << vis_delay << " ms";
        }

        const bool need_octomap_clouds =
            octomap_params_.publish_markers || octomap_params_.publish_binary;
        if (grid_params_.publish_split_clouds || need_octomap_clouds) {
            sensor_msgs::msg::PointCloud2 pc_msg;
            pcl::toROSMsg(*cloud, pc_msg);
            pc_msg.header = header;
            pc_msg.header.frame_id = fusion_output_body_frame_;
            grid_point_cloud_pub_->publish(pc_msg);

            sensor_msgs::msg::PointCloud2 g_msg, ng_msg;
            pcl::toROSMsg(*ground, g_msg);
            g_msg.header = header; g_msg.header.frame_id = fusion_output_body_frame_;
            grid_ground_point_cloud_pub_->publish(g_msg);
            pcl::toROSMsg(*non_ground, ng_msg);
            ng_msg.header = header; ng_msg.header.frame_id = fusion_output_body_frame_;
            grid_non_ground_point_cloud_pub_->publish(ng_msg);
        }
    }

    // 注意：原来的 grid_create_freespace_mask 和 grid_create_freespace_label_map 已不再需要，已删除

    std::vector<cv::Point> grid_extract_polygon_points(const std::vector<float>& bbox, const cv::Size& image_size) {
        std::vector<cv::Point> points;
        int w = image_size.width, h = image_size.height;
        int exp = grid_params_.expand_pixels;
        for (size_t i = 0; i < bbox.size(); i += 2) {
            int x = static_cast<int>(std::round(bbox[i]));
            int y = static_cast<int>(std::round(bbox[i+1]));
            points.emplace_back(x, y);
        }
        if (exp == 0) return points;
        cv::Point2f center(0, 0);
        for (const auto& p : points) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x /= points.size();
        center.y /= points.size();
        std::vector<cv::Point> expanded;
        expanded.reserve(points.size());
        for (const auto& p : points) {
            float dx = p.x - center.x;
            float dy = p.y - center.y;
            float len = std::sqrt(dx*dx + dy*dy);
            if (len < 1e-6) {
                expanded.push_back(p);
                continue;
            }
            int new_x = static_cast<int>(std::round(p.x + dx / len * exp));
            int new_y = static_cast<int>(std::round(p.y + dy / len * exp));
            new_x = std::max(0, std::min(w-1, new_x));
            new_y = std::max(0, std::min(h-1, new_y));
            expanded.emplace_back(new_x, new_y);
        }
        return expanded;
    }

    std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr,
           pcl::PointCloud<pcl::PointXYZ>::Ptr,
           pcl::PointCloud<pcl::PointXYZ>::Ptr,
           nav_msgs::msg::OccupancyGrid>
    grid_generate_point_cloud_and_grid(const cv::Mat& depth_image, const cv::Mat& mask,
                                                    const cv::Mat& label_map,
                                                    const std_msgs::msg::Header& header, bool depth_only_mode,
                                                    std::vector<cv::Point2f>& out_proj_points,
                                                    std::vector<int>& out_labels)
    {
        PerformanceTimer timer;
        timer.start();
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground(new pcl::PointCloud<pcl::PointXYZ>);
        nav_msgs::msg::OccupancyGrid grid = grid_initialize_occupancy_grid(header);

        int w = depth_image.cols, h = depth_image.rows;
        const float* data = reinterpret_cast<const float*>(depth_image.data);

        struct TempPoint {
            pcl::PointXYZ pt;
            bool is_ground;
            cv::Point2f img_pt;
            int label;
        };
        std::vector<TempPoint> temp_points;
        temp_points.reserve((w / grid_params_.skip_pixels) * (h / grid_params_.skip_pixels));

        // 统计边缘填充点数（可选）
        size_t edge_filled_count = 0;

        for (int v = 0; v < h; v += grid_params_.skip_pixels) {
            for (int u = 0; u < w; u += grid_params_.skip_pixels) {
                float depth = data[v * w + u] * grid_params_.depth_scale;
                if (depth < grid_params_.depth_min || depth > grid_params_.depth_max) continue;

                pcl::PointXYZ point = grid_calculate_3d_point(u, v, depth);
                pcl::PointXYZ transformed = grid_transform_point(point);
                if (!grid_is_point_in_valid_range(transformed)) continue;

                // 从 label_map 获取类别标签
                int label = label_map.at<int>(v, u);
                bool is_ground = false;

                // ========== 新增：图像边缘强制视为可通行区域 ==========
                bool is_edge = (u < grid_params_.expand_pixels ||
                                u >= w - grid_params_.expand_pixels ||
                                v < grid_params_.expand_pixels ||
                                v >= h - grid_params_.expand_pixels);
                if (is_edge) {
                    // 边缘像素直接标记为地面（可通行），并将标签强制设为0（freespace）
                    is_ground = true;
                    label = 0;   // 边缘强制为可通行类别（freespace）
                    edge_filled_count++;
                } else {
                    // 非边缘区域根据语义标签判断
                    auto it = class_map_.find(label);
                    if (it != class_map_.end()) {
                        const std::string& class_name = it->second;
                        if (std::find(freespace_classes_list_.begin(), freespace_classes_list_.end(), class_name) != freespace_classes_list_.end()) {
                            is_ground = true;
                        }
                    } else {
                        // 未知类别，视为非地面（障碍物）
                        is_ground = false;
                        LOG(INFO) << "Unknown label " << label << " at pixel (" << u << "," << v << ")";
                    }
                }
                // ====================================================

                temp_points.push_back({transformed, is_ground, cv::Point2f(u, v), label});
            }
        }

        // 可选的日志输出边缘填充统计
        if (edge_filled_count > 0) {
            LOG(INFO) << "[DepthToGrid] Edge-filled points count: " << edge_filled_count;
        }

        size_t points_before_filter = temp_points.size();
        // 可选的密度滤波（原代码中已有，此处保留）
        if (filter_enabled_) {
            PerformanceTimer filter_timer;
            filter_timer.start();

            std::vector<size_t> roi_indices;
            for (size_t i = 0; i < temp_points.size(); ++i) {
                const auto& tp = temp_points[i];
                if (!tp.is_ground &&
                    tp.pt.x >= filter_roi_x_min_ && tp.pt.x <= filter_roi_x_max_ &&
                    tp.pt.y >= filter_roi_y_min_ && tp.pt.y <= filter_roi_y_max_ &&
                    tp.pt.z >= filter_roi_z_min_ && tp.pt.z <= filter_roi_z_max_) {
                    roi_indices.push_back(i);
                }
            }

            if (!roi_indices.empty()) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                roi_cloud->reserve(roi_indices.size());
                for (size_t idx : roi_indices) {
                    const auto& tp = temp_points[idx];
                    pcl::PointXYZI p;
                    p.x = tp.pt.x; p.y = tp.pt.y; p.z = tp.pt.z;
                    p.intensity = static_cast<float>(idx);
                    roi_cloud->push_back(p);
                }

                pcl::RadiusOutlierRemoval<pcl::PointXYZI> radius_filter;
                radius_filter.setInputCloud(roi_cloud);
                radius_filter.setRadiusSearch(filter_radius_);
                radius_filter.setMinNeighborsInRadius(filter_min_neighbors_);
                radius_filter.setKeepOrganized(false);
                pcl::PointCloud<pcl::PointXYZI> filtered_roi;
                radius_filter.filter(filtered_roi);

                std::vector<bool> is_in_roi(temp_points.size(), false);
                for (size_t idx : roi_indices) is_in_roi[idx] = true;

                std::vector<bool> keep(temp_points.size(), false);
                for (size_t i = 0; i < temp_points.size(); ++i) {
                    if (!is_in_roi[i]) keep[i] = true;
                }
                for (const auto& p : filtered_roi.points) {
                    size_t idx = static_cast<size_t>(p.intensity);
                    if (idx < temp_points.size()) keep[idx] = true;
                }

                std::vector<TempPoint> filtered_temp;
                filtered_temp.reserve(temp_points.size());
                for (size_t i = 0; i < temp_points.size(); ++i) {
                    if (keep[i]) filtered_temp.push_back(temp_points[i]);
                }
                temp_points.swap(filtered_temp);
                double filter_elapsed = filter_timer.elapsedMs();
                LOG(INFO) << "[Depth Filter] ROI range x=[" << filter_roi_x_min_ << "," << filter_roi_x_max_
                        << "], y=[" << filter_roi_y_min_ << "," << filter_roi_y_max_
                        << "], z=[" << filter_roi_z_min_ << "," << filter_roi_z_max_
                        << "], radius=" << filter_radius_ << ", min_neighbors=" << filter_min_neighbors_
                        << ", points_before=" << roi_indices.size() << ", points_after=" << filtered_roi.size()
                        << ", total_before=" << points_before_filter << ", total_after=" << temp_points.size()
                        << ", took " << filter_elapsed << " ms";
            } else {
                LOG(INFO) << "[Depth Filter] No points in ROI, skipping filter";
            }
        }

        // 先处理地面点
        for (const auto& tp : temp_points) {
            if (!tp.is_ground) continue;
            int ix = static_cast<int>((tp.pt.x - grid.info.origin.position.x) / grid_params_.grid_resolution);
            int iy = static_cast<int>((tp.pt.y - grid.info.origin.position.y) / grid_params_.grid_resolution);
            if (ix >= 0 && ix < grid_params_.grid_size && iy >= 0 && iy < grid_params_.grid_size) {
                int idx = iy * grid_params_.grid_size + ix;
                if (tp.label == 255) {
                    grid.data[idx] = 100;
                } 
                else if (tp.label == 0) {
                    grid.data[idx] = 0;
                }
                else {
                    grid.data[idx] = tp.label + 49;
                }
            }
        }
        // 再处理非地面点
        for (const auto& tp : temp_points) {
            if (tp.is_ground) continue;
            int ix = static_cast<int>((tp.pt.x - grid.info.origin.position.x) / grid_params_.grid_resolution);
            int iy = static_cast<int>((tp.pt.y - grid.info.origin.position.y) / grid_params_.grid_resolution);
            if (ix >= 0 && ix < grid_params_.grid_size && iy >= 0 && iy < grid_params_.grid_size) {
                int idx = iy * grid_params_.grid_size + ix;
                if (tp.label == 255) {
                    grid.data[idx] = 100;
                } 
                else if (tp.label == 0) {
                    grid.data[idx] = 0;
                }
                else {
                    grid.data[idx] = tp.label + 49;
                }
            }
        }


        // 填充点云用于可视化输出
        cloud->reserve(temp_points.size());
        ground->reserve(temp_points.size());
        non_ground->reserve(temp_points.size());
        out_proj_points.reserve(temp_points.size());
        out_labels.reserve(temp_points.size());
        for (const auto& tp : temp_points) {
            cloud->push_back(tp.pt);
            if (tp.is_ground) ground->push_back(tp.pt);
            else non_ground->push_back(tp.pt);
            out_proj_points.push_back(tp.img_pt);
            out_labels.push_back(tp.label);
        }

        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[DepthToGrid] Generated " << cloud->size() << " points, ground=" << ground->size()
                << ", non_ground=" << non_ground->size() << ", took " << elapsed << " ms";
        return std::make_tuple(cloud, ground, non_ground, grid);
    }

    void grid_generate_from_pointcloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& pc_msg,
        const std_msgs::msg::Header& header,
        nav_msgs::msg::OccupancyGrid& grid,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground)
    {
        PerformanceTimer timer;
        timer.start();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*pc_msg, *cloud);
        grid = grid_initialize_occupancy_grid(header);
        ground->clear(); non_ground->clear();

        struct TempPoint {
            pcl::PointXYZ pt;
            bool is_ground;
        };
        std::vector<TempPoint> temp_points;
        temp_points.reserve(cloud->size());

        for (const auto& pt : cloud->points) {
            if (!pcl::isFinite(pt)) continue;
            if (pt.x < grid_params_.point_cloud_range_min_x || pt.x > grid_params_.point_cloud_range_max_x ||
                pt.y < grid_params_.point_cloud_range_min_y || pt.y > grid_params_.point_cloud_range_max_y ||
                pt.z < grid_params_.point_cloud_range_min_z || pt.z > grid_params_.point_cloud_range_max_z) continue;

            bool is_ground = ((pt.z > grid_params_.freespace_mini_z && pt.z < grid_params_.freespace_min_z) ||
                              (pt.z > grid_params_.freespace_max_z) ||
                              (pt.x >= grid_params_.ground_region_x_min_ && pt.x <= grid_params_.ground_region_x_max_ &&
                               pt.y >= grid_params_.ground_region_y_min_ && pt.y <= grid_params_.ground_region_y_max_));

            if (is_ground) ground->push_back(pt);
            else non_ground->push_back(pt);
            temp_points.push_back({pt, is_ground});
        }

        for (const auto& tp : temp_points) {
            if (tp.is_ground) {
                int ix = static_cast<int>((tp.pt.x - grid.info.origin.position.x) / grid_params_.grid_resolution);
                int iy = static_cast<int>((tp.pt.y - grid.info.origin.position.y) / grid_params_.grid_resolution);
                if (ix >= 0 && ix < grid_params_.grid_size && iy >= 0 && iy < grid_params_.grid_size) {
                    grid.data[iy * grid_params_.grid_size + ix] = 0;
                }
            }
        }
        for (const auto& tp : temp_points) {
            if (!tp.is_ground) {
                int ix = static_cast<int>((tp.pt.x - grid.info.origin.position.x) / grid_params_.grid_resolution);
                int iy = static_cast<int>((tp.pt.y - grid.info.origin.position.y) / grid_params_.grid_resolution);
                if (ix >= 0 && ix < grid_params_.grid_size && iy >= 0 && iy < grid_params_.grid_size) {
                    grid.data[iy * grid_params_.grid_size + ix] = 100;
                }
            }
        }
        double elapsed = timer.elapsedMs();
        LOG(INFO) << "[PointCloudToGrid] Input points=" << cloud->size() << ", ground=" << ground->size()
                  << ", non_ground=" << non_ground->size() << ", took " << elapsed << " ms";
    }

    void drawProjectedPoints(cv::Mat& rgb_image,
        const std::vector<cv::Point2f>& points,
        const std::vector<int>& labels,
        const std::vector<std::string>& /*classes*/)
    {
        if (rgb_image.empty() || points.empty()) return;

        // 静态缓存：为每个语义标签分配固定颜色（只初始化一次）
        static std::unordered_map<int, cv::Scalar> label_color_cache;
        static std::once_flag color_init_flag;
        std::call_once(color_init_flag, [this]() {
        // 定义颜色池（BGR 顺序），用于不同类别
        std::vector<cv::Scalar> color_pool = {
        cv::Scalar(0, 255, 0),     // 绿色 (可通行 freespace)
        cv::Scalar(0, 165, 255),   // 橙色 (stair)
        cv::Scalar(255, 0, 255),   // 洋红 (shallow grassland)
        cv::Scalar(0, 0, 255),     // 红色 (curb)
        cv::Scalar(255, 255, 0),   // 青色 (escalator)
        cv::Scalar(128, 0, 128),   // 紫色
        cv::Scalar(255, 0, 0),     // 蓝色
        cv::Scalar(0, 255, 255)    // 黄色
        };
        int color_idx = 0;
        for (const auto& kv : class_map_) {
        int label = kv.first;
        if (label == 255) {
        // 背景用白色
        label_color_cache[label] = cv::Scalar(255, 255, 255);
        } else {
        cv::Scalar color = color_pool[color_idx % color_pool.size()];
        label_color_cache[label] = color;
        color_idx++;
        }
        }
        // 为不在 class_map_ 中的标签设置默认颜色（灰色）
        label_color_cache[-1] = cv::Scalar(128, 128, 128);
        });

        // 绘制每个投影点
        for (size_t i = 0; i < points.size(); ++i) {
        const auto& pt = points[i];
        int label = labels[i];
        cv::Scalar color;
        auto it = label_color_cache.find(label);
        if (it != label_color_cache.end()) {
        color = it->second;
        } else {
        color = label_color_cache[-1];  // 灰色
        }
        cv::circle(rgb_image, pt, 2, color, -1);
        }
    }
    
    void publishProjectedPointsImage(const cv::Mat& image, const std_msgs::msg::Header& header) {
        if (!projected_points_pub_ || image.empty()) return;
        sensor_msgs::msg::Image ros_img;
        cv_bridge::CvImage(header, "bgr8", image).toImageMsg(ros_img);
        ros_img.header.frame_id = "camera";
        projected_points_pub_->publish(ros_img);
        double stamp = ros_img.header.stamp.sec + ros_img.header.stamp.nanosec * 1e-9;
        double delay = (this->now().seconds() - stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Output] Projected points image published, stamp=" << stamp
                  << ", delay=" << delay << " ms";
    }

    pcl::PointXYZ grid_calculate_3d_point(int u, int v, float depth) {
        pcl::PointXYZ pt;
        pt.x = (u - grid_params_.cx) * depth / grid_params_.fx;
        pt.y = (v - grid_params_.cy) * depth / grid_params_.fy;
        pt.z = depth;
        return pt;
    }

    pcl::PointXYZ grid_transform_point(const pcl::PointXYZ& pt) {
        Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0);
        p = grid_transformation_matrix_ * p;
        return pcl::PointXYZ(p(0), p(1), p(2));
    }

    bool grid_is_point_in_valid_range(const pcl::PointXYZ& pt) {
        return (pt.x >= grid_params_.point_cloud_range_min_x && pt.x <= grid_params_.point_cloud_range_max_x &&
                pt.y >= grid_params_.point_cloud_range_min_y && pt.y <= grid_params_.point_cloud_range_max_y &&
                pt.z >= grid_params_.point_cloud_range_min_z && pt.z <= grid_params_.point_cloud_range_max_z);
    }

    nav_msgs::msg::OccupancyGrid grid_initialize_occupancy_grid(const std_msgs::msg::Header& header) {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header = header;
        grid.header.frame_id = fusion_output_body_frame_;
        float cx = grid_params_.grid_size * grid_params_.grid_resolution / 2.0f;
        float cy = grid_params_.grid_size * grid_params_.grid_resolution / 2.0f;
        grid.info.width = grid_params_.grid_size;
        grid.info.height = grid_params_.grid_size;
        grid.info.resolution = grid_params_.grid_resolution;
        grid.info.origin.position.x = -cx;
        grid.info.origin.position.y = -cy;
        grid.info.origin.position.z = 0.0;
        grid.data.resize(grid_params_.grid_size * grid_params_.grid_size, 0);
        return grid;
    }

    void grid_update_time_statistics(double callback_time_ms) {
        grid_total_callback_count_++;
        grid_total_callback_time_ += callback_time_ms;
        if (callback_time_ms > grid_max_callback_time_) grid_max_callback_time_ = callback_time_ms;
        if (callback_time_ms < grid_min_callback_time_) grid_min_callback_time_ = callback_time_ms;
        if (grid_total_callback_count_ % grid_statistics_interval_ == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - grid_last_statistics_time_).count();
            LOG(INFO) << "Grid stats: avg=" << (grid_total_callback_time_/grid_total_callback_count_)
                      << "ms max=" << grid_max_callback_time_ << "ms min=" << grid_min_callback_time_
                      << "ms over " << grid_statistics_interval_ << " calls (" << secs << "s)";
            grid_last_statistics_time_ = now;
        }
    }

    // -------------------- OctoMap + 2D 栅格记忆模块 --------------------
    struct OctomapParameters {
        std::string octomap_local_markers_topic;
        std::string octomap_local_binary_topic;
        std::string octomap_global_markers_topic;
        std::string octomap_global_binary_topic;
        double resolution;
        double max_range;
        double time_tolerance;
        bool publish_binary;
        bool publish_markers;
        bool enable_profiling;
        int sync_queue_size;
        bool enable_accumulation;
        std::string odom_topic;
        std::string global_frame;
        std::string local_frame;
        bool downsample_enabled;
        double voxel_leaf_size;
        bool compute_free_space;
        // 记忆栅格参数
        double mem_body_front;      // 机体系前方记忆距离（m）
        double mem_body_rear;       // 机体系后方记忆距离（m）
        double mem_body_left;       // 机体系左侧记忆距离（m）
        double mem_body_right;      // 机体系右侧记忆距离（m）
        double memory_radius;       // map 系最大记忆水平距离，超出则清除
        double memory_clear_interval_s; // 噪点定时清除间隔（s），0 表示不清
    } octomap_params_;

    std::shared_ptr<octomap::OcTree> local_octree_;
    std::shared_ptr<octomap::OcTree> global_octree_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr octomap_local_marker_pub_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_local_binary_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr octomap_global_marker_pub_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_global_binary_pub_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> octomap_ground_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> octomap_non_ground_sub_;
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> OctomapSyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<OctomapSyncPolicy>> octomap_sync_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    double last_memory_clear_time_ = 0.0;

    struct OctomapProfilingData {
        double total_time = 0.0;
        double sync_check_time = 0.0;
        double ground_processing_time = 0.0;
        double local_marker_publish_time = 0.0;
        size_t ground_points_count = 0;
        size_t non_ground_points_count = 0;
    };

    double getCurrentTimeMs() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
    }

    void octomap_initialize() {
        local_octree_ = std::make_shared<octomap::OcTree>(octomap_params_.resolution);
        global_octree_ = std::make_shared<octomap::OcTree>(octomap_params_.resolution);

        octomap_local_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            octomap_params_.octomap_local_markers_topic, 10);
        octomap_local_binary_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>(
            octomap_params_.octomap_local_binary_topic, 10);
        octomap_global_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            octomap_params_.octomap_global_markers_topic, 10);
        octomap_global_binary_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>(
            octomap_params_.octomap_global_binary_topic, 10);

        octomap_ground_sub_.subscribe(this, grid_params_.ground_pointcloud_topic);
        octomap_non_ground_sub_.subscribe(this, grid_params_.non_ground_pointcloud_topic);

        octomap_sync_ = std::make_shared<message_filters::Synchronizer<OctomapSyncPolicy>>(
            OctomapSyncPolicy(octomap_params_.sync_queue_size), octomap_ground_sub_, octomap_non_ground_sub_);
        octomap_sync_->registerCallback(std::bind(&GridMapTask::octomap_sync_callback, this,
                                                   std::placeholders::_1, std::placeholders::_2));

        if (octomap_params_.enable_accumulation) {
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                octomap_params_.odom_topic, rclcpp::QoS(1),
                std::bind(&GridMapTask::odom_callback, this, std::placeholders::_1));
            LOG(INFO) << "Odometry subscribed for accumulation: " << octomap_params_.odom_topic;
        }

        LOG(INFO) << "OctoMap module initialized with resolution: " << octomap_params_.resolution << " m";
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double stamp = rclcpp::Time(msg->header.stamp).seconds();
        double system_now = this->now().seconds();
        double delay_ms = (system_now - stamp) * 1000.0;
        LOG(INFO) << std::setprecision(13) << "[Input] Odom received, stamp=" << stamp
                  << ", system_time=" << system_now << ", delay=" << delay_ms << " ms";
        
        grid_map::DataPool* DP = grid_map::DataPool::Instance();
        grid_map::OdomData odom_data;
        odom_data.time = stamp;
        odom_data.pose = msg->pose.pose;
        DP->main_data_.odom_cache.Push(odom_data);
    }

    Eigen::Affine3d poseToAffine(const geometry_msgs::msg::Pose& pose) const {
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() << pose.position.x, pose.position.y, pose.position.z;
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
                             pose.orientation.y, pose.orientation.z);
        transform.linear() = q.matrix();
        return transform;
    }

    // ==================== 3D voxel hash 栅格记忆（map 系） ====================
    struct VoxelKey { int gx; int gy; int gz; };
    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& k) const {
            size_t h = std::hash<int>()(k.gx);
            h ^= std::hash<int>()(k.gy) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.gz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct VoxelKeyEqual {
        bool operator()(const VoxelKey& a, const VoxelKey& b) const {
            return a.gx == b.gx && a.gy == b.gy && a.gz == b.gz;
        }
    };
    // value: 100=正障碍（未来可扩展 -100=负障碍等）
    std::unordered_map<VoxelKey, int8_t, VoxelKeyHash, VoxelKeyEqual> memory_voxel_map_;

    int worldToGrid(double w, double res) const {
        return static_cast<int>(std::floor(w / res));
    }

    // 从 non_ground 点云中提取机体近域的 3D 点 → 变换到 map → 写入 voxel hash
    size_t memoryInsertFromCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
                                  const geometry_msgs::msg::Pose& pose) {
        if (!non_ground || non_ground->empty()) return 0;
        const Eigen::Affine3d T = poseToAffine(pose);
        const double front = octomap_params_.mem_body_front;
        const double rear  = octomap_params_.mem_body_rear;
        const double left  = octomap_params_.mem_body_left;
        const double right = octomap_params_.mem_body_right;
        const double res = octomap_params_.resolution;  // voxel 分辨率
        size_t inserted = 0;
        for (const auto& p : non_ground->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            // 机体近域过滤
            if (p.x > front || p.x < -rear) continue;
            if (p.y > left  || p.y < -right) continue;
            // 变换到 map 系
            const Eigen::Vector3d pw = T * Eigen::Vector3d(p.x, p.y, p.z);
            VoxelKey key{worldToGrid(pw.x(), res),
                         worldToGrid(pw.y(), res),
                         worldToGrid(pw.z(), res)};
            memory_voxel_map_[key] = 100;
            ++inserted;
        }
        return inserted;
    }

    // 清除 map 系超出记忆窗口的 voxel（水平距离）
    size_t memoryPruneOutside(const geometry_msgs::msg::Pose& pose) {
        const double rx = pose.position.x;
        const double ry = pose.position.y;
        const double mem_r2 = octomap_params_.memory_radius * octomap_params_.memory_radius;
        const double res = octomap_params_.resolution;
        size_t pruned = 0;
        for (auto it = memory_voxel_map_.begin(); it != memory_voxel_map_.end(); ) {
            double wx = (it->first.gx + 0.5) * res;
            double wy = (it->first.gy + 0.5) * res;
            double dx = wx - rx;
            double dy = wy - ry;
            if (dx * dx + dy * dy > mem_r2) {
                it = memory_voxel_map_.erase(it);
                ++pruned;
            } else {
                ++it;
            }
        }
        return pruned;
    }

    // 3D 记忆 voxel → 投影到 body 2D 栅格（列投影取 max）合并到当前帧 grid
    void memoryMergeToBody(nav_msgs::msg::OccupancyGrid& body_grid,
                           const geometry_msgs::msg::Pose& pose) {
        if (memory_voxel_map_.empty()) return;
        const Eigen::Affine3d T_inv = poseToAffine(pose).inverse();
        const double vres = octomap_params_.resolution;
        const double gres = grid_params_.grid_resolution;
        const int W = grid_params_.grid_size;
        const double ox = body_grid.info.origin.position.x;
        const double oy = body_grid.info.origin.position.y;
        for (const auto& kv : memory_voxel_map_) {
            double wx = (kv.first.gx + 0.5) * vres;
            double wy = (kv.first.gy + 0.5) * vres;
            double wz = (kv.first.gz + 0.5) * vres;
            Eigen::Vector3d pb = T_inv * Eigen::Vector3d(wx, wy, wz);
            int ix = static_cast<int>((pb.x() - ox) / gres);
            int iy = static_cast<int>((pb.y() - oy) / gres);
            if (ix < 0 || ix >= W || iy < 0 || iy >= W) continue;
            int idx = iy * W + ix;
            if (body_grid.data[idx] < kv.second) {
                body_grid.data[idx] = kv.second;
            }
        }
    }

    // 定时全量清除（防噪点累积）
    bool memoryCheckPeriodicClear(double system_now_s) {
        if (octomap_params_.memory_clear_interval_s <= 0.0) return false;
        if (last_memory_clear_time_ <= 0.0) {
            last_memory_clear_time_ = system_now_s;
            return false;
        }
        if (system_now_s - last_memory_clear_time_ >= octomap_params_.memory_clear_interval_s) {
            size_t old_size = memory_voxel_map_.size();
            memory_voxel_map_.clear();
            last_memory_clear_time_ = system_now_s;
            LOG(INFO) << "[Memory] Periodic clear: removed " << old_size << " voxels";
            return true;
        }
        return false;
    }

    // 保留：OctoMap → 2D（用于不开记忆时的 OctoMap 单帧）
    nav_msgs::msg::OccupancyGrid octomapToOccupancyGrid(
        const std::shared_ptr<octomap::OcTree>& octree,
        const std_msgs::msg::Header& header)
    {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header = header;
        grid.header.frame_id = fusion_output_body_frame_;

        float cx = grid_params_.grid_size * grid_params_.grid_resolution / 2.0f;
        float cy = grid_params_.grid_size * grid_params_.grid_resolution / 2.0f;
        grid.info.width = grid_params_.grid_size;
        grid.info.height = grid_params_.grid_size;
        grid.info.resolution = grid_params_.grid_resolution;
        grid.info.origin.position.x = -cx;
        grid.info.origin.position.y = -cy;
        grid.info.origin.position.z = 0.0;
        grid.info.origin.orientation.w = 1.0;
        grid.data.assign(grid_params_.grid_size * grid_params_.grid_size, 0);

        for (octomap::OcTree::leaf_iterator it = octree->begin_leafs(), end = octree->end_leafs(); it != end; ++it) {
            double x = it.getX();
            double y = it.getY();
            int ix = static_cast<int>((x - grid.info.origin.position.x) / grid_params_.grid_resolution);
            int iy = static_cast<int>((y - grid.info.origin.position.y) / grid_params_.grid_resolution);
            if (ix < 0 || ix >= static_cast<int>(grid_params_.grid_size) ||
                iy < 0 || iy >= static_cast<int>(grid_params_.grid_size)) continue;
            int idx = iy * grid_params_.grid_size + ix;
            if (octree->isNodeOccupied(*it)) {
                grid.data[idx] = 100;
            } else {
                if (grid.data[idx] != 100) grid.data[idx] = 0;
            }
        }
        return grid;
    }

    void octomap_sync_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& ground_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& non_ground_msg)
    {
        // 记忆不依赖 OctoMap；板端关闭 marker/binary 时直接跳过（避免 fromROSMsg + OcTree）
        if (!octomap_params_.publish_markers && !octomap_params_.publish_binary) {
            return;
        }

        PerformanceTimer total_timer;
        total_timer.start();
        OctomapProfilingData prof;
        double start_time = octomap_params_.enable_profiling ? getCurrentTimeMs() : 0.0;
        double step_start = start_time;

        double ground_stamp = rclcpp::Time(ground_msg->header.stamp).seconds();
        double non_ground_stamp = rclcpp::Time(non_ground_msg->header.stamp).seconds();
        double system_now = this->now().seconds();
        LOG(INFO) << std::setprecision(13) << "[OctoMap Input] Ground stamp=" << ground_stamp
                  << ", Non-ground stamp=" << non_ground_stamp
                  << ", system_time=" << system_now;

        if (octomap_params_.enable_profiling) {
            prof.sync_check_time = getCurrentTimeMs() - step_start;
            step_start = getCurrentTimeMs();
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pcl(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_pcl(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*ground_msg, *ground_pcl);
        pcl::fromROSMsg(*non_ground_msg, *non_ground_pcl);

        if (octomap_params_.enable_profiling) {
            prof.ground_points_count = ground_pcl->size();
            prof.non_ground_points_count = non_ground_pcl->size();
        }

        // 记忆已移到 grid_publish_results；OctoMap 回调只做单帧可视化/binary
        local_octree_->clear();
        if (octomap_params_.compute_free_space) {
            for (const auto& point : ground_pcl->points) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
                double range = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
                if (range <= octomap_params_.max_range) {
                    local_octree_->updateNode(octomap::point3d(point.x, point.y, point.z), false);
                }
            }
        }
        for (const auto& point : non_ground_pcl->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
            double range = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            if (range <= octomap_params_.max_range) {
                local_octree_->updateNode(octomap::point3d(point.x, point.y, point.z), true);
            }
        }
        local_octree_->updateInnerOccupancy();

        if (octomap_params_.enable_profiling) {
            prof.ground_processing_time = getCurrentTimeMs() - step_start;
            step_start = getCurrentTimeMs();
        }

        if (octomap_params_.publish_markers) {
            octomap_publish_markers(local_octree_, ground_msg->header, octomap_local_marker_pub_,
                                    octomap_params_.local_frame, prof.local_marker_publish_time);
        }
        if (octomap_params_.publish_binary) {
            octomap_publish_octomap(local_octree_, ground_msg->header, octomap_local_binary_pub_,
                                    octomap_params_.local_frame);
        }

        if (octomap_params_.enable_profiling) {
            prof.local_marker_publish_time = getCurrentTimeMs() - step_start;
            prof.total_time = getCurrentTimeMs() - start_time;
            LOG(INFO) << "=== OctoMap Performance (single-frame) ===";
            LOG(INFO) << "Total: " << prof.total_time << " ms, Sync: " << prof.sync_check_time
                      << ", Ground(" << prof.ground_points_count << "): " << prof.ground_processing_time
                      << ", LocalMarker: " << prof.local_marker_publish_time;
        }
        double total_elapsed = total_timer.elapsedMs();
        LOG(INFO) << "[Timing] OctoMap callback total time: " << total_elapsed << " ms";
    }

    void octomap_publish_markers(const std::shared_ptr<octomap::OcTree>& octree,
                                 const std_msgs::msg::Header& header,
                                 const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub,
                                 const std::string& frame_id,
                                 double& time_ms)
    {
        PerformanceTimer timer;
        timer.start();
        visualization_msgs::msg::MarkerArray marker_array;

        visualization_msgs::msg::Marker free_marker;
        free_marker.header = header;
        free_marker.header.frame_id = frame_id;
        free_marker.ns = "octomap_free";
        free_marker.id = 0;
        free_marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
        free_marker.action = visualization_msgs::msg::Marker::ADD;
        free_marker.scale.x = octomap_params_.resolution;
        free_marker.scale.y = octomap_params_.resolution;
        free_marker.scale.z = octomap_params_.resolution;
        free_marker.color.r = 0.0;
        free_marker.color.g = 1.0;
        free_marker.color.b = 0.0;
        free_marker.color.a = 0.3;
        free_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

        visualization_msgs::msg::Marker occupied_marker = free_marker;
        occupied_marker.ns = "octomap_occupied";
        occupied_marker.id = 1;
        occupied_marker.color.r = 1.0;
        occupied_marker.color.g = 0.0;
        occupied_marker.color.b = 0.0;
        occupied_marker.color.a = 0.8;

        size_t free_count = 0, occupied_count = 0;
        for (octomap::OcTree::iterator it = octree->begin(), end = octree->end(); it != end; ++it) {
            geometry_msgs::msg::Point cube_center;
            cube_center.x = it.getX();
            cube_center.y = it.getY();
            cube_center.z = it.getZ();
            if (octree->isNodeOccupied(*it)) {
                occupied_marker.points.push_back(cube_center);
                occupied_count++;
            } else {
                free_marker.points.push_back(cube_center);
                free_count++;
            }
        }

        if (!free_marker.points.empty()) marker_array.markers.push_back(free_marker);
        if (!occupied_marker.points.empty()) marker_array.markers.push_back(occupied_marker);

        pub->publish(marker_array);
        double elapsed = timer.elapsedMs();
        time_ms = elapsed;
        double stamp = header.stamp.sec + header.stamp.nanosec * 1e-9;
        double sys_now = this->now().seconds();
        LOG(INFO) << std::setprecision(13) << "[OctoMap Output] Markers published in " << frame_id
                  << ", stamp=" << stamp << ", system_time=" << sys_now << ", delay=" << (sys_now - stamp)*1000.0
                  << " ms, free=" << free_count << ", occupied=" << occupied_count << ", took " << elapsed << " ms";
    }

    void octomap_publish_octomap(const std::shared_ptr<octomap::OcTree>& octree,
                                 const std_msgs::msg::Header& header,
                                 const rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr& pub,
                                 const std::string& frame_id)
    {
        PerformanceTimer timer;
        timer.start();
        octomap_msgs::msg::Octomap msg;
        msg.header = header;
        msg.header.frame_id = frame_id;
        msg.binary = octomap_params_.publish_binary;
        msg.id = "OcTree";
        msg.resolution = octomap_params_.resolution;

        if (octomap_msgs::fullMapToMsg(*octree, msg)) {
            pub->publish(msg);
            double elapsed = timer.elapsedMs();
            double stamp = header.stamp.sec + header.stamp.nanosec * 1e-9;
            double sys_now = this->now().seconds();
            LOG(INFO) << std::setprecision(13) << "[OctoMap Output] Octomap published in " << frame_id
                      << ", stamp=" << stamp << ", system_time=" << sys_now << ", delay=" << (sys_now - stamp)*1000.0
                      << " ms, binary=" << msg.binary << ", took " << elapsed << " ms";
        } else {
            LOG(ERROR) << "Failed to convert OctoMap to message";
        }
    }

    // -------------------- 统一参数加载 --------------------
    bool loadParameters() {
        try {
            const char* root_dir = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
            if (!root_dir) {
                LOG(ERROR) << "SERES_PERCEPTION_INSTALL_DIR environment variable not set";
                return false;
            }
            std::string config_path = std::string(root_dir) + "/grid_map_memory_node/cfg/grid_map_memory_node.yaml";
            LOG(INFO) << " config_path: " << config_path;
            YAML::Node config = YAML::LoadFile(config_path);

            const bool new_layout = static_cast<bool>(config["pointcloud"]) && static_cast<bool>(config["occupancy_grid"]);
            if (new_layout) {
                if (!loadParametersV2(config)) return false;
            } else {
                LOG(WARNING) << "Using legacy YAML layout (depth_lidar_fusion / vision_freespace_process / octomap)";
                if (!loadParametersLegacy(config)) return false;
            }

            LOG(INFO) << "Parameters loaded successfully (layout=" << (new_layout ? "v2" : "legacy") << ")";
            LOG(INFO) << "[cfg] mode=" << fusion_mode_
                      << " camera fx/fy/cx/cy=" << fusion_fx_ << "/" << fusion_fy_ << "/"
                      << fusion_cx_ << "/" << fusion_cy_
                      << " (grid_" << grid_params_.fx << "/" << grid_params_.fy << "/"
                      << grid_params_.cx << "/" << grid_params_.cy << ")";
            LOG(INFO) << "[cfg] memory.enable=" << octomap_params_.enable_accumulation
                      << " keep_radius=" << octomap_params_.memory_radius
                      << " body F/R/L/R=" << octomap_params_.mem_body_front << "/"
                      << octomap_params_.mem_body_rear << "/"
                      << octomap_params_.mem_body_left << "/" << octomap_params_.mem_body_right;
            LOG(INFO) << "[cfg] qos reliability=" << sensor_qos_reliability_
                      << " history_depth=" << sensor_qos_history_depth_
                      << " durability=" << sensor_qos_durability_;
            return true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Parameter load exception: " << e.what();
            return false;
        }
    }

    void applyLidarToBodyExtrinsic(double body_rx, double body_ry, double body_rz,
                                   double body_tx, double body_ty, double body_tz) {
        Eigen::Matrix3d R_body;
        R_body = Eigen::AngleAxisd(body_rz, Eigen::Vector3d::UnitZ())
               * Eigen::AngleAxisd(body_ry, Eigen::Vector3d::UnitY())
               * Eigen::AngleAxisd(body_rx, Eigen::Vector3d::UnitX());
        fusion_intermediate_to_body_.setIdentity();
        fusion_intermediate_to_body_.block<3,3>(0,0) = R_body;
        fusion_intermediate_to_body_.block<3,1>(0,3) << body_tx, body_ty, body_tz;
    }

    void buildExpandClassLabels() {
        freespace_classes_list_ = grid_params_.freespace_classes;
        grid_params_.expand_class_labels.clear();
        std::map<std::string, int> class_name_to_label;
        for (const auto& pair : class_map_) {
            class_name_to_label[pair.second] = pair.first;
        }
        for (const auto& class_name : grid_params_.expand_classes) {
            auto it = class_name_to_label.find(class_name);
            if (it != class_name_to_label.end()) {
                grid_params_.expand_class_labels.push_back(it->second);
                LOG(INFO) << "Class '" << class_name << "' will be dilated (label=" << it->second << ")";
            } else {
                LOG(WARNING) << "Expand class '" << class_name << "' not found in class mapping, ignored";
            }
        }
    }

    bool loadParametersV2(const YAML::Node& config) {
        fusion_mode_ = config["mode"].as<std::string>("depth_only");
        loadSensorQosFromYaml(config);

        auto topics = config["topics"];
        if (!topics) { LOG(ERROR) << "Missing topics"; return false; }
        fusion_depth_image_topic_ = topics["depth"].as<std::string>();
        grid_params_.depth_topic = fusion_depth_image_topic_;
        grid_params_.depth_image_type = topics["depth_image_type"].as<std::string>("raw");
        fusion_camera_info_topic_ = topics["camera_info"].as<std::string>("");
        fusion_lidar_topic_ = topics["lidar"].as<std::string>("/livox/lidar");
        grid_params_.rgb_topic = topics["rgb"].as<std::string>("");
        grid_params_.rgb_image_type = topics["rgb_image_type"].as<std::string>("raw");
        grid_params_.mask_topic = topics["mask"].as<std::string>("/segment/mask_raw");
        octomap_params_.odom_topic = topics["odom"].as<std::string>("/Odometry");
        grid_params_.fused_pointcloud_topic = topics["fused_pointcloud"].as<std::string>("/fused_point_cloud");
        filter_raw_topic_ = topics["fused_pointcloud_raw"].as<std::string>("/fused_point_cloud_raw");
        grid_params_.occupancy_grid_topic = topics["occupancy_grid"].as<std::string>("/map/occupancy_grid");
        grid_params_.pointcloud_topic = topics["source_pointcloud"].as<std::string>("/grid/source_point_cloud");
        grid_params_.ground_pointcloud_topic = topics["ground_points"].as<std::string>("/grid/source_ground_points");
        grid_params_.non_ground_pointcloud_topic = topics["non_ground_points"].as<std::string>("/grid/source_non_ground_points");
        grid_params_.visualization_topic = topics["visualization"].as<std::string>("/camera/vis_freespace_seg");
        grid_params_.projected_points_vis_topic = topics["projected_points_vis"].as<std::string>("/grid/projected_points_vis");

        auto cam = config["camera"];
        if (!cam) { LOG(ERROR) << "Missing camera"; return false; }
        fusion_fx_ = cam["fx"].as<double>();
        fusion_fy_ = cam["fy"].as<double>();
        fusion_cx_ = cam["cx"].as<double>();
        fusion_cy_ = cam["cy"].as<double>();
        grid_params_.fx = fusion_fx_;
        grid_params_.fy = fusion_fy_;
        grid_params_.cx = fusion_cx_;
        grid_params_.cy = fusion_cy_;
        grid_params_.depth_scale = cam["depth_scale"].as<double>(0.001);

        auto ext = config["extrinsic"];
        if (!ext) { LOG(ERROR) << "Missing extrinsic"; return false; }
        auto c2l = ext["camera_to_lidar"];
        fusion_rotation_x_ = c2l["rotation_x"].as<double>(-1.5708);
        fusion_rotation_y_ = c2l["rotation_y"].as<double>(0.0);
        fusion_rotation_z_ = c2l["rotation_z"].as<double>(-1.5708);
        fusion_translation_x_ = c2l["translation_x"].as<double>(0.0);
        fusion_translation_y_ = c2l["translation_y"].as<double>(0.0);
        fusion_translation_z_ = c2l["translation_z"].as<double>(0.0);
        auto l2b = ext["lidar_to_body"];
        applyLidarToBodyExtrinsic(
            l2b["rotation_x"].as<double>(0.0), l2b["rotation_y"].as<double>(0.0), l2b["rotation_z"].as<double>(0.0),
            l2b["translation_x"].as<double>(0.0), l2b["translation_y"].as<double>(0.0), l2b["translation_z"].as<double>(0.0));
        fusion_intermediate_frame_ = ext["intermediate_frame"].as<std::string>("livox_frame");
        fusion_output_body_frame_ = ext["body_frame"].as<std::string>("base_link");
        auto c2b = ext["camera_to_body"];
        grid_params_.yaw = c2b["yaw"].as<double>(-1.5708);
        grid_params_.pitch = c2b["pitch"].as<double>(1.5708);
        grid_params_.roll = c2b["roll"].as<double>(-1.5708);
        grid_params_.pitch2 = c2b["pitch2"].as<double>(0.0);
        grid_params_.x_offset = c2b["x"].as<double>(0.0);
        grid_params_.y_offset = c2b["y"].as<double>(0.0);
        grid_params_.z_offset = c2b["z"].as<double>(0.0);

        auto pc = config["pointcloud"];
        fusion_depth_sample_step_ = pc["depth_sample_step"].as<int>(6);
        fusion_depth_min_threshold_ = pc["depth_min_m"].as<double>(-0.01);
        fusion_depth_max_threshold_ = pc["depth_max_m"].as<double>(5.0);
        fusion_time_sync_threshold_ = pc["time_sync_threshold_s"].as<double>(0.30);
        fusion_max_queue_size_ = pc["max_queue_size"].as<int>(5);
        auto filter_node = pc["filter"];
        if (filter_node) {
            filter_enabled_ = filter_node["enabled"].as<bool>(false);
            filter_roi_x_min_ = filter_node["roi_x_min"].as<double>(-0.8);
            filter_roi_x_max_ = filter_node["roi_x_max"].as<double>(0.8);
            filter_roi_y_min_ = filter_node["roi_y_min"].as<double>(-0.8);
            filter_roi_y_max_ = filter_node["roi_y_max"].as<double>(0.8);
            filter_roi_z_min_ = filter_node["roi_z_min"].as<double>(-0.8);
            filter_roi_z_max_ = filter_node["roi_z_max"].as<double>(0.8);
            filter_radius_ = filter_node["radius"].as<double>(0.1);
            filter_min_neighbors_ = filter_node["min_neighbors"].as<int>(10);
        } else {
            filter_enabled_ = false;
        }

        auto og = config["occupancy_grid"];
        grid_params_.time_sync_threshold = og["time_sync_threshold_s"].as<double>(0.10);
        grid_params_.max_queue_size = og["max_queue_size"].as<int>(10);
        grid_params_.depth_min = og["depth_min_m"].as<double>(0.00001);
        grid_params_.depth_max = og["depth_max_m"].as<double>(5.0);
        grid_params_.skip_pixels = og["skip_pixels"].as<int>(6);
        grid_params_.point_cloud_range_min_x = og["range_x_min"].as<double>(-5.0);
        grid_params_.point_cloud_range_max_x = og["range_x_max"].as<double>(5.0);
        grid_params_.point_cloud_range_min_y = og["range_y_min"].as<double>(-5.0);
        grid_params_.point_cloud_range_max_y = og["range_y_max"].as<double>(5.0);
        grid_params_.point_cloud_range_min_z = og["range_z_min"].as<double>(-1.0);
        grid_params_.point_cloud_range_max_z = og["range_z_max"].as<double>(5.0);
        grid_params_.freespace_max_z = og["freespace_max_z"].as<double>(1.5);
        grid_params_.freespace_min_z = og["freespace_min_z"].as<double>(-0.2);
        grid_params_.freespace_mini_z = og["freespace_mini_z"].as<double>(-0.6);
        grid_params_.freespace_classes = og["freespace_classes"]
            ? og["freespace_classes"].as<std::vector<std::string>>()
            : std::vector<std::string>{"freespace"};
        grid_params_.expand_pixels = og["expand_pixels"].as<int>(8);
        grid_params_.freespace_boundary_line_width = og["boundary_dilate_width"].as<int>(2);
        if (og["expand_classes"]) {
            grid_params_.expand_classes = og["expand_classes"].as<std::vector<std::string>>();
        } else {
            grid_params_.expand_classes.clear();
        }
        grid_params_.ground_region_x_min_ = og["ground_region_x_min"].as<double>(-0.3);
        grid_params_.ground_region_x_max_ = og["ground_region_x_max"].as<double>(0.7);
        grid_params_.ground_region_y_min_ = og["ground_region_y_min"].as<double>(-0.3);
        grid_params_.ground_region_y_max_ = og["ground_region_y_max"].as<double>(0.3);
        grid_params_.grid_resolution = og["resolution"].as<float>(0.05f);
        grid_params_.grid_size = og["size"].as<int>(200);
        grid_params_.enable_fusion_with_pointcloud = og["enable_lidar_grid_fusion"].as<bool>(false);
        grid_params_.publish_debug_vis = og["publish_debug_vis"].as<bool>(false);
        grid_params_.publish_split_clouds = og["publish_split_clouds"].as<bool>(false);
        grid_params_.lidar_only_mode = (fusion_mode_ == "lidar_only");

        class_map_.clear();
        if (og["class_mapping"]) {
            for (const auto& item : og["class_mapping"]) {
                class_map_[std::stoi(item.first.as<std::string>())] = item.second.as<std::string>();
            }
        } else {
            class_map_[0] = "freespace";
            class_map_[255] = "background";
        }
        buildExpandClassLabels();

        auto mem = config["memory"];
        if (mem) {
            octomap_params_.enable_accumulation = mem["enable"].as<bool>(false);
            octomap_params_.resolution = mem["voxel_resolution"].as<double>(0.05);
            octomap_params_.mem_body_front = mem["body_front_m"].as<double>(2.0);
            octomap_params_.mem_body_rear = mem["body_rear_m"].as<double>(2.0);
            octomap_params_.mem_body_left = mem["body_left_m"].as<double>(1.5);
            octomap_params_.mem_body_right = mem["body_right_m"].as<double>(1.5);
            octomap_params_.memory_radius = mem["keep_radius_m"].as<double>(5.0);
            octomap_params_.memory_clear_interval_s = mem["clear_interval_s"].as<double>(30.0);
            octomap_params_.time_tolerance = mem["odom_time_tolerance_s"].as<double>(0.2);
        } else {
            octomap_params_.enable_accumulation = false;
            octomap_params_.resolution = 0.05;
        }

        auto viz = config["octomap_viz"];
        if (viz) {
            octomap_params_.publish_binary = viz["publish_binary"].as<bool>(false);
            octomap_params_.publish_markers = viz["publish_markers"].as<bool>(false);
            octomap_params_.enable_profiling = viz["enable_profiling"].as<bool>(true);
            octomap_params_.sync_queue_size = viz["sync_queue_size"].as<int>(10);
            octomap_params_.max_range = viz["max_range_m"].as<double>(3.0);
            octomap_params_.downsample_enabled = viz["downsample_enabled"].as<bool>(false);
            octomap_params_.voxel_leaf_size = viz["voxel_leaf_size"].as<double>(0.1);
            octomap_params_.compute_free_space = viz["compute_free_space"].as<bool>(false);
            octomap_params_.local_frame = viz["local_frame"].as<std::string>("base_link");
            octomap_params_.global_frame = viz["global_frame"].as<std::string>("map");
            octomap_params_.octomap_local_markers_topic = viz["local_markers_topic"].as<std::string>("/octomap/local_markers");
            octomap_params_.octomap_local_binary_topic = viz["local_binary_topic"].as<std::string>("/octomap/local_binary");
            octomap_params_.octomap_global_markers_topic = viz["global_markers_topic"].as<std::string>("/octomap/global_markers");
            octomap_params_.octomap_global_binary_topic = viz["global_binary_topic"].as<std::string>("/octomap/global_binary");
        } else {
            octomap_params_.publish_binary = false;
            octomap_params_.publish_markers = false;
            octomap_params_.enable_profiling = true;
            octomap_params_.sync_queue_size = 10;
            octomap_params_.max_range = 3.0;
        }

        LOG(INFO) << "[v2] mode=" << fusion_mode_
                  << " depth=" << fusion_depth_image_topic_
                  << " mask=" << grid_params_.mask_topic
                  << " odom=" << octomap_params_.odom_topic
                  << " memory=" << octomap_params_.enable_accumulation;
        return true;
    }

    // 兼容旧版 depth_lidar_fusion / vision_freespace_process / octomap 三段式 yaml
    bool loadParametersLegacy(const YAML::Node& config) {
        loadSensorQosFromYaml(config);
        auto fusion_node = config["depth_lidar_fusion"]["ros__parameters"];
        if (!fusion_node) {
            LOG(ERROR) << "Missing depth_lidar_fusion parameters";
            return false;
        }
        fusion_mode_ = fusion_node["mode"].as<std::string>("fusion");
        fusion_time_sync_threshold_ = fusion_node["time_sync_threshold"].as<double>(0.2);
        fusion_max_queue_size_ = fusion_node["max_queue_size"].as<int>(10);
        fusion_rotation_x_ = fusion_node["depth_to_lidar_rotation_x"].as<double>(-1.5708);
        fusion_rotation_y_ = fusion_node["depth_to_lidar_rotation_y"].as<double>(0.0);
        fusion_rotation_z_ = fusion_node["depth_to_lidar_rotation_z"].as<double>(-1.5708);
        fusion_translation_x_ = fusion_node["depth_to_lidar_translation_x"].as<double>(0.52);
        fusion_translation_y_ = fusion_node["depth_to_lidar_translation_y"].as<double>(-0.05);
        fusion_translation_z_ = fusion_node["depth_to_lidar_translation_z"].as<double>(-0.15);
        fusion_intermediate_frame_ = fusion_node["output_frame"].as<std::string>("lidar");
        fusion_output_body_frame_ = fusion_node["output_body_frame"].as<std::string>(fusion_intermediate_frame_);
        applyLidarToBodyExtrinsic(
            fusion_node["output_body_rotation_x"].as<double>(0.0),
            fusion_node["output_body_rotation_y"].as<double>(0.0),
            fusion_node["output_body_rotation_z"].as<double>(0.0),
            fusion_node["output_body_translation_x"].as<double>(0.0),
            fusion_node["output_body_translation_y"].as<double>(0.0),
            fusion_node["output_body_translation_z"].as<double>(0.0));
        fusion_depth_image_topic_ = fusion_node["depth_image_topic"].as<std::string>();
        fusion_camera_info_topic_ = fusion_node["camera_info_topic"].as<std::string>("");
        fusion_lidar_topic_ = fusion_node["lidar_pointcloud_topic"].as<std::string>("/livox/lidar");
        fusion_fx_ = fusion_node["camera_fx"].as<double>(382.604);
        fusion_fy_ = fusion_node["camera_fy"].as<double>(382.328);
        fusion_cx_ = fusion_node["camera_cx"].as<double>(319.230);
        fusion_cy_ = fusion_node["camera_cy"].as<double>(176.157);
        fusion_depth_sample_step_ = fusion_node["depth_sample_step"].as<int>(4);
        fusion_depth_min_threshold_ = fusion_node["depth_min_threshold"].as<double>(-0.01);
        fusion_depth_max_threshold_ = fusion_node["depth_max_threshold"].as<double>(20.0);
        auto filter_node = fusion_node["filter"];
        if (filter_node) {
            filter_enabled_ = filter_node["enabled"].as<bool>(false);
            filter_roi_x_min_ = filter_node["roi_x_min"].as<double>(-0.8);
            filter_roi_x_max_ = filter_node["roi_x_max"].as<double>(0.8);
            filter_roi_y_min_ = filter_node["roi_y_min"].as<double>(-0.8);
            filter_roi_y_max_ = filter_node["roi_y_max"].as<double>(0.8);
            filter_roi_z_min_ = filter_node["roi_z_min"].as<double>(-0.8);
            filter_roi_z_max_ = filter_node["roi_z_max"].as<double>(0.8);
            filter_radius_ = filter_node["radius"].as<double>(0.1);
            filter_min_neighbors_ = filter_node["min_neighbors"].as<int>(10);
            filter_raw_topic_ = filter_node["raw_topic"].as<std::string>("/fused_point_cloud_raw");
        } else {
            filter_enabled_ = false;
            filter_raw_topic_ = "/fused_point_cloud_raw";
        }

        auto grid_node = config["vision_freespace_process"];
        if (!grid_node) {
            LOG(ERROR) << "Missing vision_freespace_process parameters";
            return false;
        }
        auto topics = grid_node["topics"];
        if (topics) {
            grid_params_.rgb_topic = topics["rgb_topic"].as<std::string>("");
            grid_params_.depth_topic = topics["depth_topic"].as<std::string>(fusion_depth_image_topic_);
            grid_params_.mask_topic = topics["mask_topic"].as<std::string>("/segment/mask_raw");
            grid_params_.pointcloud_topic = topics["pointcloud_topic"].as<std::string>("/grid/source_point_cloud");
            grid_params_.occupancy_grid_topic = topics["occupancy_grid_topic"].as<std::string>("/map/occupancy_grid");
            grid_params_.visualization_topic = topics["visualization_topic"].as<std::string>("/camera/vis_freespace_seg");
            grid_params_.ground_pointcloud_topic = topics["ground_pointcloud_topic"].as<std::string>("/grid/source_ground_points");
            grid_params_.non_ground_pointcloud_topic = topics["non_ground_pointcloud_topic"].as<std::string>("/grid/source_non_ground_points");
            grid_params_.fused_pointcloud_topic = topics["fused_pointcloud_topic"].as<std::string>("/fused_point_cloud");
            grid_params_.projected_points_vis_topic = topics["projected_points_vis_topic"].as<std::string>("/grid/projected_points_vis");
            grid_params_.rgb_image_type = topics["rgb_image_type"].as<std::string>("compressed");
            grid_params_.depth_image_type = topics["depth_image_type"].as<std::string>("raw");
        }
        auto intrinsics = grid_node["camera_intrinsics"];
        if (intrinsics) {
            grid_params_.fx = intrinsics["fx"].as<double>(fusion_fx_);
            grid_params_.fy = intrinsics["fy"].as<double>(fusion_fy_);
            grid_params_.cx = intrinsics["cx"].as<double>(fusion_cx_);
            grid_params_.cy = intrinsics["cy"].as<double>(fusion_cy_);
        } else {
            grid_params_.fx = fusion_fx_; grid_params_.fy = fusion_fy_;
            grid_params_.cx = fusion_cx_; grid_params_.cy = fusion_cy_;
        }
        auto trans = grid_node["transform_params"];
        if (trans) {
            grid_params_.yaw = trans["yaw"].as<double>(-1.5708);
            grid_params_.pitch = trans["pitch"].as<double>(1.5708);
            grid_params_.roll = trans["roll"].as<double>(-1.5708);
            grid_params_.pitch2 = trans["pitch2"].as<double>(0.0);
            grid_params_.x_offset = trans["x_offset"].as<double>(0.0);
            grid_params_.y_offset = trans["y_offset"].as<double>(0.0);
            grid_params_.z_offset = trans["z_offset"].as<double>(0.0);
        }
        auto proc = grid_node["processing_params"];
        if (proc) {
            grid_params_.max_queue_size = proc["max_queue_size"].as<int>(10);
            grid_params_.time_sync_threshold = proc["time_sync_threshold"].as<double>(0.30);
            grid_params_.depth_scale = proc["depth_scale"].as<double>(0.001);
            grid_params_.depth_min = proc["depth_min"].as<double>(0.00001);
            grid_params_.depth_max = proc["depth_max"].as<double>(100.0);
            grid_params_.skip_pixels = proc["skip_pixels"].as<int>(3);
            grid_params_.point_cloud_range_min_x = proc["point_cloud_range_min_x"].as<double>(-5.0);
            grid_params_.point_cloud_range_max_x = proc["point_cloud_range_max_x"].as<double>(5.0);
            grid_params_.point_cloud_range_min_y = proc["point_cloud_range_min_y"].as<double>(-5.0);
            grid_params_.point_cloud_range_max_y = proc["point_cloud_range_max_y"].as<double>(5.0);
            grid_params_.point_cloud_range_min_z = proc["point_cloud_range_min_z"].as<double>(-1.0);
            grid_params_.point_cloud_range_max_z = proc["point_cloud_range_max_z"].as<double>(5.0);
            grid_params_.freespace_max_z = proc["freespace_max_z"].as<double>(1.5);
            grid_params_.freespace_min_z = proc["freespace_min_z"].as<double>(-0.2);
            grid_params_.freespace_mini_z = proc["freespace_mini_z"].as<double>(-0.6);
            grid_params_.freespace_classes = proc["freespace_classes"]
                ? proc["freespace_classes"].as<std::vector<std::string>>()
                : std::vector<std::string>{"freespace"};
            grid_params_.expand_pixels = proc["expand_pixels"].as<int>(8);
            grid_params_.freespace_boundary_line_width = proc["freespace_boundary_line_width"].as<int>(2);
            if (proc["expand_classes"]) {
                grid_params_.expand_classes = proc["expand_classes"].as<std::vector<std::string>>();
            }
            grid_params_.ground_region_x_min_ = proc["ground_region_x_min_"].as<double>(-0.3);
            grid_params_.ground_region_x_max_ = proc["ground_region_x_max_"].as<double>(0.7);
            grid_params_.ground_region_y_min_ = proc["ground_region_y_min_"].as<double>(-0.3);
            grid_params_.ground_region_y_max_ = proc["ground_region_y_max_"].as<double>(0.3);
        }
        auto grid_p = grid_node["grid_params"];
        if (grid_p) {
            grid_params_.grid_resolution = grid_p["resolution"].as<float>(0.05f);
            grid_params_.grid_size = grid_p["size"].as<int>(200);
        }
        auto fusion_p = grid_node["fusion_params"];
        grid_params_.enable_fusion_with_pointcloud = fusion_p
            ? fusion_p["enable_fusion_with_pointcloud"].as<bool>(false) : false;
        grid_params_.publish_debug_vis = grid_node["publish_debug_vis"].as<bool>(false);
        grid_params_.publish_split_clouds = grid_node["publish_split_clouds"].as<bool>(false);
        grid_params_.lidar_only_mode = (fusion_mode_ == "lidar_only");

        class_map_.clear();
        if (grid_node["class_mapping"]) {
            for (const auto& item : grid_node["class_mapping"]) {
                class_map_[std::stoi(item.first.as<std::string>())] = item.second.as<std::string>();
            }
        } else {
            class_map_[0] = "freespace";
            class_map_[255] = "background";
        }
        buildExpandClassLabels();

        auto octomap_node = config["octomap"];
        if (octomap_node) {
            octomap_params_.octomap_local_markers_topic = octomap_node["octomap_local_markers_topic"].as<std::string>("/octomap/local_markers");
            octomap_params_.octomap_local_binary_topic = octomap_node["octomap_local_binary_topic"].as<std::string>("/octomap/local_binary");
            octomap_params_.octomap_global_markers_topic = octomap_node["octomap_global_markers_topic"].as<std::string>("/octomap/global_markers");
            octomap_params_.octomap_global_binary_topic = octomap_node["octomap_global_binary_topic"].as<std::string>("/octomap/global_binary");
            octomap_params_.resolution = octomap_node["resolution"].as<double>(0.05);
            octomap_params_.max_range = octomap_node["max_range"].as<double>(3.0);
            octomap_params_.time_tolerance = octomap_node["time_tolerance"].as<double>(0.2);
            octomap_params_.publish_binary = octomap_node["publish_binary"].as<bool>(false);
            octomap_params_.publish_markers = octomap_node["publish_markers"].as<bool>(false);
            octomap_params_.enable_profiling = octomap_node["enable_profiling"].as<bool>(true);
            octomap_params_.sync_queue_size = octomap_node["sync_queue_size"].as<int>(10);
            octomap_params_.enable_accumulation = octomap_node["enable_accumulation"].as<bool>(false);
            octomap_params_.odom_topic = octomap_node["odom_topic"].as<std::string>("/Odometry");
            octomap_params_.global_frame = octomap_node["global_frame"].as<std::string>("map");
            octomap_params_.local_frame = octomap_node["local_frame"].as<std::string>("base_link");
            octomap_params_.downsample_enabled = octomap_node["downsample_enabled"].as<bool>(false);
            octomap_params_.voxel_leaf_size = octomap_node["voxel_leaf_size"].as<double>(0.1);
            octomap_params_.compute_free_space = octomap_node["compute_free_space"].as<bool>(false);
            octomap_params_.mem_body_front = octomap_node["mem_body_front"].as<double>(2.0);
            octomap_params_.mem_body_rear = octomap_node["mem_body_rear"].as<double>(2.0);
            octomap_params_.mem_body_left = octomap_node["mem_body_left"].as<double>(1.5);
            octomap_params_.mem_body_right = octomap_node["mem_body_right"].as<double>(1.5);
            octomap_params_.memory_radius = octomap_node["memory_radius"].as<double>(5.0);
            octomap_params_.memory_clear_interval_s = octomap_node["memory_clear_interval_s"].as<double>(30.0);
        } else {
            octomap_params_.enable_accumulation = false;
            octomap_params_.resolution = 0.05;
            octomap_params_.publish_binary = false;
            octomap_params_.publish_markers = false;
            octomap_params_.odom_topic = "/Odometry";
        }
        return true;
    }

};


void GridMapTask::processPendingInputs() {
    grid_map::DataPool* DP = grid_map::DataPool::Instance();
    if (fusion_mode_ == "depth_only") {
        grid_map::FusionDepthData depth_data;
        int discarded = 0;
        if (DP->main_data_.fusion_depth_cache.PopLatestByStamp(
                depth_data, [](const grid_map::FusionDepthData& d) { return d.time; },
                &discarded) == 0) {
            if (discarded > 0) {
                LOG(INFO) << "[Process] depth_cache skipped " << discarded
                          << " stale frame(s), stamp=" << std::setprecision(13) << depth_data.time;
            }
            processDepthForGrid(depth_data.msg, nullptr, nullptr);
        }
    } else if (fusion_mode_ == "lidar_only") {
        grid_map::FusionLidarData lidar_data;
        int discarded = 0;
        if (DP->main_data_.fusion_lidar_cache.PopLatestByStamp(
                lidar_data, [](const grid_map::FusionLidarData& d) { return d.time; },
                &discarded) == 0) {
            if (discarded > 0) {
                LOG(INFO) << "[Process] lidar_cache skipped " << discarded
                          << " stale frame(s), stamp=" << std::setprecision(13) << lidar_data.time;
            }
            processLidarOnlyFrame(lidar_data.msg);
        }
    } else {
        fusion_tryTimeSync();
    }
}


int main(int argc, char** argv)
{
    grid_map::Logger::Init("/tmp/grid_map_logs", "grid_map_memory_node");

    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<GridMapTask>();
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        rclcpp::WallRate loop_rate(50);
        while (rclcpp::ok()) {
            executor.spin_some();
            node->processPendingInputs();
            loop_rate.sleep();
        }
        executor.remove_node(node);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception: " << e.what();
        return 1;
    }
    rclcpp::shutdown();
    grid_map::Logger::Shutdown();
    return 0;
}