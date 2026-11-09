#ifndef SERES_PERCEPTION_VISUALPERCEPTION_COMMON_TYPE_DEF_H__
#define SERES_PERCEPTION_VISUALPERCEPTION_COMMON_TYPE_DEF_H__


#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <list>
#include <mutex>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
namespace seres{
namespace perception {

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

  int PopNewest(T &data) {
    std::unique_lock<std::mutex> locker(mutex_);
    RemoveUnvalidData();
    if (data_cache.size() <= 0) return -1;
    data = data_cache.back().data;
    data_cache.pop_back();
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
      std::cout << " ... data time : " << time << std::endl;
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


typedef struct Det2D {
  int label;
  float confidence;
  float img_box_x;
  float img_box_y;
  float img_box_h;
  float img_box_w;
} Det2D;

typedef struct Seg2D {
  int label;
  float confidence;
  std::vector<float> mask_x;
  std::vector<float> mask_y;
} Seg2D;


typedef struct
{
	double time;
  std::string img_name ;
  std::vector<Det2D> det_results;
  std::vector<Seg2D> seg_results;
} DetSegResults;

typedef struct ImageData {
  double time;
  std::string img_name = "none";
  cv::Mat img;
} ImageData;


typedef struct PerceptionMainData {
  DataCache<ImageData> img_cache;
  DataCache<DetSegResults> detseg_cache;

  PerceptionMainData() {

    img_cache.setConfig(1e100, 500, 500, 0);
    detseg_cache.setConfig(1e100, 500, 500, 0);
  }
} PerceptionMainData;

}
}
#endif
