#ifndef SERES_PERCEPTION_3D_MAP_DATA_POOL_H_
#define SERES_PERCEPTION_3D_MAP_DATA_POOL_H_

// c++ common header
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <pcl_conversions/pcl_conversions.h>
#include "common_type.h"
#include <iostream>

#define BASE_DECLARE_SINGLETON(classname)                                      \
 public:                                                                  \
  static classname *Instance() {                                          \
    static classname *instance = nullptr;                                 \
    if (!instance) {                                                      \
      static std::once_flag flag;                                         \
      std::call_once(flag,                                                \
                     [&] { instance = new (std::nothrow) classname(); }); \
    }                                                                     \
    return instance;                                                      \
  }                                                                       \
                                                                          \
 private:                                                                 \
  classname(const classname &) = delete;                                  \
  classname &operator=(const classname &) = delete;



namespace seres {
namespace perception {



class DataPool {
public:
  DataPool() = default; // Add this default constructor
  PerceptionMainData* GetMainDataPtr() { return &main_data_; }
  PerceptionMainData& GetMainDataRef() { return main_data_; }

  // int BakLidarProcessThreadData() {
  //   // if (main_data_.top_lidar_data_cache.PopNewest(lidar_process_thread_data_.lidar_raw_data.point_cloud_ptr_map[0]) < 0) {
  //   //   return 1;
  //   // }
  //   return 0;
  // }

  // void PushRSResult() {
  //   std::unique_lock<std::mutex> locker(lidar_mutex_);

  //   // std::cout  << "PushLidarResult(), lidar2 timestamp: " << std::setprecision(13) << main_data_.lidar_process_thread_data_.lidar_output.time << std::endl;
  // }

public:
  std::mutex lidar_mutex_;
  PerceptionMainData main_data_;

  BASE_DECLARE_SINGLETON(DataPool)
};


} // namespace perception
} // namespace seres

#endif