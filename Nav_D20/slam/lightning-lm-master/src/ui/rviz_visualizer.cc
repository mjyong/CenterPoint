#include "ui/rviz_visualizer.h"

#include <deque>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "wrapper/ros_utils.h"

namespace lightning::ui {

struct RvizVisualizer::Impl {
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_world_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_global_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyframes_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_keyframe_poses_;

    nav_msgs::msg::Path path_;
    std::string frame_id_{"map"};
    std::string child_frame_id_{"body_link"};

    SE3 T_imu_lidar_ = SE3();
    int current_scan_size_ = 10;
    int max_keyframes_ = 200;
    std::deque<sensor_msgs::msg::PointCloud2> scan_msgs_;
    std::vector<Keyframe::Ptr> keyframes_;

    Impl(rclcpp::Node::SharedPtr node) : node_(node) {
        if (node_) {
            pub_scan_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/current_scan", 10);
            pub_scan_world_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/current_scan_world", 10);
            //pub_global_map_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/global_map", 2);
            pub_keyframes_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/keyframes", 2);
            pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/lio_path", 10);
            pub_odom_ = node_->create_publisher<nav_msgs::msg::Odometry>("/lightning/odom", 10);
            pub_keyframe_poses_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/keyframe_path", 10);
        }
        path_.header.frame_id = frame_id_;
    }

    void PublishScan(CloudPtr cloud, const SE3& pose) {
        if (!cloud || cloud->empty()) return;
        if (!node_) return;

        // 发布扫描点云（激光雷达坐标系）50
        sensor_msgs::msg::PointCloud2 scan_msg;
        pcl::toROSMsg(*cloud, scan_msg);
        scan_msg.header.stamp = node_->now();
        scan_msg.header.frame_id = child_frame_id_;
        if (pub_scan_) pub_scan_->publish(scan_msg);

        // 转换到世界坐标系并发布
        CloudPtr cloud_world(new PointCloudType);
        pcl::transformPointCloud(*cloud, *cloud_world, pose.matrix().cast<float>());
        
        sensor_msgs::msg::PointCloud2 scan_world_msg;
        pcl::toROSMsg(*cloud_world, scan_world_msg);
        scan_world_msg.header.stamp = scan_msg.header.stamp;
        scan_world_msg.header.frame_id = frame_id_;
        if (pub_scan_world_) pub_scan_world_->publish(scan_world_msg);

        // 保留最近的点云用于显示（可选）
        scan_msgs_.push_back(scan_world_msg);
        if (scan_msgs_.size() > current_scan_size_) {
            scan_msgs_.pop_front();
        }
    }

    void PublishNavState(const NavState& state) {
        if (!node_) return;
        // 发布里程计
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = node_->now();
        odom.header.frame_id = frame_id_;
        odom.child_frame_id = child_frame_id_;

        odom.pose.pose.position.x = state.pos_.x();
        odom.pose.pose.position.y = state.pos_.y();
        odom.pose.pose.position.z = state.pos_.z();
        odom.pose.pose.orientation.x = state.rot_.unit_quaternion().x();
        odom.pose.pose.orientation.y = state.rot_.unit_quaternion().y();
        odom.pose.pose.orientation.z = state.rot_.unit_quaternion().z();
        odom.pose.pose.orientation.w = state.rot_.unit_quaternion().w();

        // 速度
        odom.twist.twist.linear.x = state.vel_.x();
        odom.twist.twist.linear.y = state.vel_.y();
        odom.twist.twist.linear.z = state.vel_.z();
        odom.twist.twist.angular.x = state.omega_b_.x();
        odom.twist.twist.angular.y = state.omega_b_.y();
        odom.twist.twist.angular.z = state.omega_b_.z();

        if (pub_odom_) pub_odom_->publish(odom);

        // 更新路径
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = odom.header;
        pose_stamped.pose = odom.pose.pose;
        path_.poses.push_back(pose_stamped);
        path_.header.stamp = odom.header.stamp;
        if (pub_path_) pub_path_->publish(path_);
    }

    void PublishGlobalMap(const std::map<int, CloudPtr>& cloud_map) {
        if (!node_) return;
        // 合并所有点云并发布（简化实现）
        CloudPtr merged(new PointCloudType);
        for (const auto& pair : cloud_map) {
            *merged += *pair.second;
        }
        if (merged->empty()) return;

        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(*merged, map_msg);
        map_msg.header.stamp = node_->now();
        map_msg.header.frame_id = frame_id_;
        if (pub_global_map_) pub_global_map_->publish(map_msg);
    }

    void PublishKeyframes() {
        if (!node_ || keyframes_.empty()) return;
        CloudPtr merged(new PointCloudType);
        nav_msgs::msg::Path keyframe_path;
        keyframe_path.header.stamp = node_->now();
        keyframe_path.header.frame_id = frame_id_;
        
        for (const auto& kf : keyframes_) {
            auto cloud = kf->GetCloud();
            SE3 pose = kf->GetOptPose();
            // 添加位姿到路径
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = node_->now();
            pose_stamped.header.frame_id = frame_id_;
            pose_stamped.pose.position.x = pose.translation().x();
            pose_stamped.pose.position.y = pose.translation().y();
            pose_stamped.pose.position.z = pose.translation().z();
            Eigen::Quaterniond q = pose.unit_quaternion();
            pose_stamped.pose.orientation.x = q.x();
            pose_stamped.pose.orientation.y = q.y();
            pose_stamped.pose.orientation.z = q.z();
            pose_stamped.pose.orientation.w = q.w();
            keyframe_path.poses.push_back(pose_stamped);
            
            // if (cloud && !cloud->empty()) {
            //     CloudPtr transformed(new PointCloudType);
            //     pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
            //     *merged += *transformed;
            // }
        }
        
        // 发布关键帧位姿路径
        if (pub_keyframe_poses_) pub_keyframe_poses_->publish(keyframe_path);
        
        //if (merged->empty()) return;
        
        // ============ 添加点云采样功能 ============
        //CloudPtr sampled(new PointCloudType);
        
        // 方案一：体素网格滤波（推荐，保持空间分布）
        // pcl::VoxelGrid<PointCloudType::PointType> voxel;
        // voxel.setInputCloud(merged);
        // voxel.setLeafSize(0.5f, 0.5f, 0.5f);  // 体素大小，可根据场景调整
        // voxel.filter(*sampled);
        
        // if (sampled->empty()) return;
        // // =========================================
        // sensor_msgs::msg::PointCloud2 kf_msg;
        // pcl::toROSMsg(*sampled, kf_msg);  // 使用采样后的点云
        // kf_msg.header.stamp = node_->now();
        // kf_msg.header.frame_id = frame_id_;
        // if (pub_keyframes_) pub_keyframes_->publish(kf_msg);
    }
};

RvizVisualizer::RvizVisualizer() = default;

RvizVisualizer::~RvizVisualizer() = default;

bool RvizVisualizer::Init(rclcpp::Node::SharedPtr node) {
    // 允许空节点（离线模式）
    impl_ = std::make_shared<Impl>(node);
    return true;
}

void RvizVisualizer::Reset(const std::vector<Keyframe::Ptr>& keyframes) {
    if (impl_) {
        impl_->path_.poses.clear();
        impl_->keyframes_.clear();
        // 可选：发布关键帧点云
    }
}

void RvizVisualizer::UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud) {
    if (impl_) {
        impl_->PublishGlobalMap(cloud);
    }
}

void RvizVisualizer::UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud) {
    // 动态点云暂不处理
}

void RvizVisualizer::UpdateNavState(const NavState& state) {
    if (impl_) {
        impl_->PublishNavState(state);
    }
}

void RvizVisualizer::UpdateRecentPose(const SE3& pose) {
    // 可合并到UpdateNavState中
}

void RvizVisualizer::UpdateScan(CloudPtr cloud, const SE3& pose) {
    if (impl_) {
        impl_->PublishScan(cloud, pose);
    }
}

void RvizVisualizer::UpdateKF(std::shared_ptr<Keyframe> kf) {
    if (impl_) {
        impl_->keyframes_.push_back(kf);
        // 限制最多保留最近的50个关键帧
        if (impl_->keyframes_.size() > impl_->max_keyframes_) {
        impl_->keyframes_.erase(impl_->keyframes_.begin());
        }
        impl_->PublishKeyframes();
    }
}

void RvizVisualizer::Quit() {
    // 无需特殊处理
}

bool RvizVisualizer::ShouldQuit() {
    return false; // rviz 不会主动退出
}

void RvizVisualizer::SetTImuLidar(const SE3& T_imu_lidar) {
    if (impl_) {
        impl_->T_imu_lidar_ = T_imu_lidar;
    }
}

void RvizVisualizer::SetCurrentScanSize(int current_scan_size) {
    if (impl_) {
        impl_->current_scan_size_ = current_scan_size;
    }
}

}  // namespace lightning::ui