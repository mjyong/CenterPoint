#ifndef GRID_MAP_COMMON_TYPE_H
#define GRID_MAP_COMMON_TYPE_H

#include <chrono>
#include <condition_variable>
#include <iterator>
#include <list>
#include <mutex>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "nav_interface/msg/det_seg_results.hpp"
#include "logging.h"
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace grid_map {

// ==================== DataCache 模板（参考 DataProcessFusionTask） ====================
template <class T>
struct TimeData {
    bool is_consumed;
    double rev_time;
    T data;
    TimeData() {
        is_consumed = false;
        rev_time = 0.0;
    }
};

template <class T>
struct DataCache {
    std::list<TimeData<T>> data_cache;
    int max_length;
    int valid_times;
    double push_time_gap_ms;
    double rm_time_gap_ms;
    std::mutex mutex_;

    DataCache() {
        data_cache.clear();
        max_length = 11;
        valid_times = 5;
        push_time_gap_ms = 500;
        rm_time_gap_ms = 10000;
    }

    void Clear() { 
        std::unique_lock<std::mutex> locker(mutex_);
        data_cache.clear(); 
    }

    void setConfig(double _rm_time_gap_ms = 10000, double _push_time_gap_ms = 500,
                   int length = 10, int _valid_times = 5) {
        if (length <= 0) length = 10;
        data_cache.clear();
        max_length = length + _valid_times;
        valid_times = _valid_times;
        push_time_gap_ms = _push_time_gap_ms;
        rm_time_gap_ms = _rm_time_gap_ms;
    }

    void Push(const T &data) {
        std::unique_lock<std::mutex> locker(mutex_);
        TimeData<T> temp;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
        temp.rev_time = duration;
        temp.data = data;
        data_cache.push_back(std::move(temp));
        while (static_cast<int>(data_cache.size()) > max_length) {
            data_cache.erase(data_cache.begin());
        }
        RemoveUnvalidData();
    }

    // 取出最新一帧并清空队列（只处理最新，丢弃积压旧帧）。
    // discarded_count：被丢弃的旧帧数（不含取出的这一帧）。
    int PopNewest(T &data, int *discarded_count = nullptr) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.size() <= 0) return -1;
        const int discarded = static_cast<int>(data_cache.size()) - 1;
        data = data_cache.back().data;
        data_cache.clear();
        if (discarded_count != nullptr) {
            *discarded_count = discarded;
        }
        return 0;
    }

    // 按传感器时间戳取最新一帧并清空其余。
    template <typename GetTimeFunc>
    int PopLatestByStamp(T &data, GetTimeFunc &&get_time, int *discarded_count = nullptr) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.size() <= 0) return -1;
        auto best_it = data_cache.begin();
        double best_stamp = get_time(best_it->data);
        for (auto it = std::next(data_cache.begin()); it != data_cache.end(); ++it) {
            const double stamp = get_time(it->data);
            if (stamp > best_stamp) {
                best_stamp = stamp;
                best_it = it;
            }
        }
        const int discarded = static_cast<int>(data_cache.size()) - 1;
        data = best_it->data;
        data_cache.clear();
        if (discarded_count != nullptr) {
            *discarded_count = discarded;
        }
        return 0;
    }

    int GetNewest(T &data) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.size() <= 0) return -1;
        data = data_cache.back().data;
        return 0;
    }

    int PopOldest(T &data) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.size() <= 0) return -1;
        data = data_cache.front().data;
        data_cache.pop_front();
        return 0;
    }

    // 基于数据时间戳查找最近的
    int GetWithinThreshold(T &data, double target_time, float threshold_ms) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.size() <= 0) return -1;
        for (auto it = data_cache.rbegin(); it != data_cache.rend(); ++it) {
            double dt_ms = std::fabs(target_time - it->data.time) * 1000;
            if (dt_ms <= threshold_ms) {
                data = it->data;
                return 0;
            }
        }
        return -1;
    }

    // 查找时间上最接近 target_time 且在阈值内的消息（遍历全部）
    int GetClosestWithinThreshold(T &data, double target_time, float threshold_ms) {
        std::unique_lock<std::mutex> locker(mutex_);
        RemoveUnvalidData();
        if (data_cache.empty()) return -1;
        
        double min_diff_ms = std::numeric_limits<double>::max();
        typename std::list<TimeData<T>>::iterator best_it = data_cache.end();
        
        for (auto it = data_cache.begin(); it != data_cache.end(); ++it) {
            double diff_ms = std::fabs(target_time - it->data.time) * 1000.0;
            if (diff_ms <= threshold_ms && diff_ms < min_diff_ms) {
                min_diff_ms = diff_ms;
                best_it = it;
            }
        }
        
        if (best_it != data_cache.end()) {
            data = best_it->data;
            return 0;
        }
        return -1;
    }

    void RemoveUnvalidData() {
        if (data_cache.size() <= 0) return;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
        auto time_now = duration;
        while ((time_now - data_cache.begin()->rev_time) * 1000 > rm_time_gap_ms) {
            data_cache.erase(data_cache.begin());
            if (data_cache.size() == 0) break;
        }
    }

    int GetDataSize() {
        std::unique_lock<std::mutex> locker(mutex_);
        return static_cast<int>(data_cache.size());
    }

    bool CheckValid() {
        if (static_cast<int>(data_cache.size()) <= valid_times) return false;
        return true;
    }
};

// ==================== GridMapTask 数据类型 ====================

// 深度图数据
struct DepthImageData {
    double time = 0.0;
    sensor_msgs::msg::Image::SharedPtr msg;
};

// RGB 压缩图数据
struct RgbImageData {
    double time = 0.0;
    sensor_msgs::msg::CompressedImage::SharedPtr msg;
};

// Freespace 分割结果数据
struct FreespaceData {
    double time = 0.0;
    nav_interface::msg::DetSegResults::SharedPtr msg;
};

// Mask 图像数据（语义分割结果）
struct MaskImageData {
    double time = 0.0;
    sensor_msgs::msg::Image::SharedPtr msg;
};

// 融合点云数据
struct FusedPointCloudData {
    double time = 0.0;
    sensor_msgs::msg::PointCloud2::SharedPtr msg;
};

// Fusion模块：深度图缓存数据
struct FusionDepthData {
    double time = 0.0;
    sensor_msgs::msg::Image::SharedPtr msg;
};

// Fusion模块：激光雷达点云缓存数据
struct FusionLidarData {
    double time = 0.0;
    sensor_msgs::msg::PointCloud2::SharedPtr msg;
};

// 里程计数据
struct OdomData {
    double time = 0.0;
    geometry_msgs::msg::Pose pose;
};

// ==================== 主数据结构 ====================
struct GridMapMainData {
    // Fusion 模块缓存
    DataCache<FusionDepthData> fusion_depth_cache;
    DataCache<FusionLidarData> fusion_lidar_cache;

    // Grid 模块缓存
    DataCache<DepthImageData> depth_image_cache;
    DataCache<RgbImageData> rgb_image_cache;
    DataCache<FreespaceData> freespace_cache;
    DataCache<MaskImageData> mask_cache;                 // 替换原 freespace_cache
    DataCache<FusedPointCloudData> fused_pointcloud_cache;

    // OctoMap 模块里程计缓存
    DataCache<OdomData> odom_cache;

    GridMapMainData() {
        // Fusion 缓存：超时10s，间隔500ms，最多20帧
        fusion_depth_cache.setConfig(10000, 500, 10, 0);
        fusion_lidar_cache.setConfig(10000, 500, 10, 0);

        // Grid 缓存
        depth_image_cache.setConfig(10000, 500, 10, 0);
        rgb_image_cache.setConfig(10000, 500, 10, 0);
        freespace_cache.setConfig(10000, 500, 10, 0);
        mask_cache.setConfig(10000, 500, 10, 0);
        fused_pointcloud_cache.setConfig(10000, 500, 10, 0);

        // 里程计缓存
        odom_cache.setConfig(10000, 500, 50, 0);
    }
};

} // namespace grid_map

#endif // GRID_MAP_COMMON_TYPE_H
