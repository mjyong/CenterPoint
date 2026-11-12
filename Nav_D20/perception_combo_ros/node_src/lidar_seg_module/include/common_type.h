#ifndef SERES_PERCEPTION_3D_MAP_COMMON_TYPE_DEF_H__
#define SERES_PERCEPTION_3D_MAP_COMMON_TYPE_DEF_H__


#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <list>
#include <mutex>
#include <vector>
#include <map>
#include "pcl/common/time.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
// #include <opencv2/opencv.hpp>
namespace seres{
namespace perception {

using PointT = pcl::PointXYZI;

#define SOURCE_LIDAR_ONLY 0
#define SOURCE_REALSENSE_ONLY 1
#define SOURCE_LIDAR_REALSENSE 2
#define SOURCE_LIDAR_VISION 3
#define SOURCE_REALSENSE_VISION 4
#define SOURCE_LIDAR_REALSENSE_VISION 5
#define SOURCE_VISION_ONLY 6
#define SOURCE_NONE 7

#define PEDESTRIAN_TYPE         0
#define UNKNOWN_TYPE            1

#define CAM_REALSENSE         0
#define CAM_LEFT              1
#define CAM_RIGHT             2

// Defines an annotated mutex that can only be locked through its scoped locker
// implementation.
class Mutex {
public:
  // A RAII class that acquires a mutex in its constructor, and
  // releases it in its destructor. It also implements waiting functionality on
  // conditions that get checked whenever the mutex is released.
  class Locker {
  public:
    Locker(Mutex *mutex) : mutex_(mutex), lock_(mutex->mutex_) {}

    ~Locker() {
      lock_.unlock();
      mutex_->condition_.notify_all();
    }

    void Await() { mutex_->condition_.wait(lock_); }

    bool AwaitWithTimeout(std::chrono::milliseconds timeout) {
      return (mutex_->condition_.wait_for(lock_, timeout) ==
              std::cv_status::no_timeout);
    }

  private:
    Mutex *mutex_;
    std::unique_lock<std::mutex> lock_;
  };

private:
  std::condition_variable condition_;
  std::mutex mutex_;
};

using MutexLocker = Mutex::Locker;

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
  void Clear() { data_cache.clear(); }

  void setConfig(double _rm_time_gap_ms = 10000, double _push_time_gap_ms = 500,
                 int length = 10, int _valid_times = 5) {
    if (length <= 0) length = 10;
    data_cache.clear();
    max_length = length + _valid_times;
    valid_times = _valid_times;
    push_time_gap_ms = _push_time_gap_ms;
    rm_time_gap_ms = _rm_time_gap_ms;
  }
  DataCache(double _rm_time_gap_ms, double _push_time_gap_ms, int length,
            int _valid_times) {
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
    while (data_cache.size() > max_length) {
      data_cache.erase(data_cache.begin());
    }

    RemoveUnvalidData();
  }

  // Pop the newest item and discard all older queued items (latest-frame-only).
  // Uses push order (back of list). Prefer PopLatestByStamp when T carries sensor time.
  // Returns 0 on success, -1 if empty. discarded_count receives number of skipped older items.
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

  // Pop the item with the maximum sensor timestamp and discard all others.
  template<typename GetTimeFunc>
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

  int GetNearestLidar(double& tar_time, T &data) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    int min_id=-1;
    double min_diff=2000;
    // std::cout<<"+++++++++tar time: "<<tar_time<< " data size:"<< data_cache.size()<<std::endl;
    for (int s = 1; s < data_cache.size(); s++) {
      double tdif = fabs(tar_time-(std::prev(data_cache.end(), s)->data->header.stamp)*0.001);
      // std::cout<<"+++++++++tar time: "<<tar_time<< "   s:"<<s<<" tdif:"<<tdif<<" stamp:"<<std::prev(data_cache.end(), s)->data->header.stamp<<std::endl;
      if(tdif<min_diff){
        min_diff=tdif;
        min_id=s;
      }
    }
    if(min_id>=0){
      data =std::prev(data_cache.end(), min_id)->data;
      return 0;
    }
    else{
      std::cout << "assign lidar failed"<<std::endl;
      return -1;
    }

  }


  int GetNearest(T &data, double time) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    // std::cout<<"+++++++++GetNearest tar time: "<<time<< " data size:"<< data_cache.size()<<std::endl;
    for (int s = 1; s < data_cache.size(); s++) {
      double t_new = std::prev(data_cache.end(), s)->data.time;
      double t_old = std::prev(data_cache.end(), (s + 1))->data.time;
      double decimal = time - (int)time;
      double time_round = (int)time + round(decimal * 10) / 10;
      std::cout << "+++++++++GetNearest time_round : " << time_round 
                << " t_new " << t_new
                << " t_old " << t_old
                << std::endl;
      if (t_new >= time_round && t_old <= time_round) {
        if ((time_round - t_old) > (t_new - time_round)) {
          data = std::prev(data_cache.end(), s)->data;
        } else {
          data = std::prev(data_cache.end(), (s + 1))->data;
        }
        // std::cout << "+++++++++GetNearest find data: " << data.time
                // << std::endl;
        return 0;

      }
    }
    data = data_cache.back().data;
    // std::cout << "+++++++++GetNearest can not find data: " << data.time
                // << std::endl;
    return 0;
  }

  int GetWithinThreshold(T &data, double target_time, float threshold_ms) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    for (int s = 1; s < data_cache.size(); s++) {
      double time = std::prev(data_cache.end(), s)->data.time;
      float dt_ms = (target_time - time) * 1000;
      // 米文时间同步： 激光-米文时间=50ms
      if (dt_ms <= threshold_ms && dt_ms > 0) {
        data = std::prev(data_cache.end(), s)->data;
        return 0;
      }
    }
    return -1;
  }

  int GetWithinThreshold(T &data, double target_time, float upper_threshold_ms, float lower_threshold_ms) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    for (int s = 1; s < data_cache.size(); s++) {
      double time = std::prev(data_cache.end(), s)->data.time;
      float dt_ms = (target_time - time) * 1000;
      if (dt_ms <= upper_threshold_ms && dt_ms >= lower_threshold_ms) {
        data = std::prev(data_cache.end(), s)->data;
        return 0;
      }
    }
    return -1;
  }

  int PopOldest(T &data) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    data = data_cache.front().data;
    data_cache.pop_front();
    return 0;
  }

  void RemoveUnvalidData() {
    if (data_cache.size() <= 0) {
      return;
    }
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
    auto time_now = duration;
    while ((time_now - data_cache.begin()->rev_time)*1000 > rm_time_gap_ms) {
      data_cache.erase(data_cache.begin());
      if (data_cache.size() == 0) break;
    }
  }

  int GetDataSize() {
    return data_cache.size();
  }

  bool CheckValid() {
    if (data_cache.size() <= valid_times) return false;
    return true;
  }
};


// 激光检测模块暂时不需要定位，融合追踪模块才需要
// typedef struct PerceptionLocType {
//   bool valid=false;
//   double time;
//   double xg = 0.0;
//   double yg = 0.0;
//   float angle = 0.0;
//   float pitch = 0.0;
//   float roll = 0.0;

// } PerceptionLocType;


struct FusionObject {
  uint32_t id = 0;      // obstacle ID.
  float x = 0.0;        // relative distance of x, VCS, unit: m
  float y = 0.0;        // relative distance of y, VCS, unit: m
  float z = 0.0;        // relative distance of z, VCS, unit: m
  float vxrel = 0.0;    // relative speed of x, VCS, unit: m/s
  float vyrel = 0.0;    // relative speed of y, VCS, unit: m/s
  double xabs = 0.0;    
  double yabs = 0.0;    
  float vxabs = 0.0;    // relative speed of x, VCS, unit: m/s
  float vyabs = 0.0;    // relative speed of y, VCS, unit: m/s
  float width = 0.0;    // unit: m
  float length = 0.0;   // unit: m
  float height = 0.0;   // unit: m
  float speed = 0.0;    // unit: m/s
  float heading = 0.0;  // unit: du
  float confidence = 0.0;
  int type = 4;
  uint32_t age = 0;
  int source;
  std::vector<int> det_in_img={0,0,0}; //顺序代表三个相机rs2 left_120 right_120
  float img_box_x = 0.0;
  float img_box_y = 0.0;
  float img_box_w = 0.0;
  float img_box_h = 0.0;
  int img_width = 0;
  int img_height = 0;
  int img_camId = -1; //三个相机rs2 left_120 right_120  0 1 2
  std::vector<int> rs2_img_coor_x;
  std::vector<int> rs2_img_coor_y;
  std::vector<double> reid_feature;
  int vision_track_id = -1;
  std::string face_id;
  bool isTrueObj = true;
};

typedef struct FaceData {
  std::string face_id;
  std::vector<int> det_in_img = {0,0,0};
  float img_box_x;
  float img_box_y;
  float img_box_w;
  float img_box_h;
  bool is_registered;
} FaceData;


typedef struct ObejctDet {
  int track_id;
  int type;
  std::vector<float> img_box;
  std::vector<float> mask_x;
  std::vector<float> mask_y;
  std::vector<float>  feature;
} ObejctDet;

typedef struct ObejctDetList {
  double time;
  std::vector<ObejctDet> obj2d;
} ObejctDetList;


typedef struct LidarDataInType {
  double time;
  pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_ptr;
  LidarDataInType() {
    time = 0.0;
    point_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>);
  }
} LidarDataInType;

typedef struct LidarDataOutType {
  double time;
  pcl::PointCloud<PointT> full_cloud;
  std::vector<int> foreground_idx;
  bool ground_points_detected;
  LidarDataOutType() {
    time = 0.0;
    full_cloud = pcl::PointCloud<PointT>();
    foreground_idx.clear();
    ground_points_detected = false;
  }
} LidarDataOutType;


typedef struct FaceDataInType {
  double time;
  std::vector<FaceData> faces;
} FaceDataInType;

typedef struct LidarProcessThreadData {
  double time;
  std::vector<FusionObject> det_rets;
} LidarProcessThreadData;

typedef struct ImageData {
  double time;
  // cv::Mat img;
} ImageData;

typedef struct LocData {
  double time;
  double xg;
  double yg;
  double heading;
  double norm_heading;
  // cv::Mat img;
} LocData;

typedef struct PerceptionMainData {

  DataCache<LocData> loc_data_cache;
  DataCache<ObejctDetList> det2d_data_cache;
  DataCache<FaceDataInType> face_data_cache;
  DataCache<LidarDataInType> lidar_data_cache;
  DataCache<LidarDataInType> front_lidar_data_cache;
  DataCache<LidarDataInType> right_lidar_data_cache;
  DataCache<LidarDataInType> left_lidar_data_cache;
  DataCache<LidarDataInType> back_lidar_data_cache;
  DataCache<LidarDataOutType> lidar_segs_cache;
  DataCache<LidarProcessThreadData> rs2_rets_cache;
  // DataCache<ImageData> img_cache;

  PerceptionMainData() {
    loc_data_cache.setConfig(1000, 500, 100, 0);
    det2d_data_cache.setConfig(1000, 500, 100, 0);
    lidar_data_cache.setConfig(1000, 500, 10, 0);
    left_lidar_data_cache.setConfig(1000, 500, 10, 0);
    right_lidar_data_cache.setConfig(1000, 500, 10, 0);
    front_lidar_data_cache.setConfig(1000, 500, 10, 0);
    back_lidar_data_cache.setConfig(1000, 500, 10, 0);
    lidar_segs_cache.setConfig(1000, 500, 10, 0);
    rs2_rets_cache.setConfig(1000, 500, 10, 0);
    // img_cache.setConfig(1000, 500, 30, 0);
    face_data_cache.setConfig(1000, 500, 30, 0);
  }
} PerceptionMainData;

}
}
#endif
