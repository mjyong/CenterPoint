

#ifndef SUPER_LIO_H_
#define SUPER_LIO_H_

#include <queue>
#include <vector>
#include <iostream>
#include <cassert>
#include <filesystem>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "basic/alias.h"
#include "common/ds.h"
#include "common/timer.h"
#include "params.h"
#include "ESKF.h"
#include "OctVoxMap/OctVoxMap.hpp"
#include "OctVoxMap/VoxelGridFilter.h"

namespace LI2Sup{

class ROSWrapper;

class SuperLIO{
public:
  using Ptr = std::shared_ptr<SuperLIO>;
  SuperLIO(){};
  ~SuperLIO(){};

  void setROSWrapper(const std::shared_ptr<ROSWrapper>& wrapper){
    data_wrapper_ = wrapper;
  }
  virtual void init();
  void process();
  void processWithMeasure(const MeasureGroup& meas);
  void saveMap();
  void printTimeRecord();

  ESKF::Ptr GetESKF() const { return kf_; }
  BASIC::CloudPtr GetScanUndistortFull() const { return scan_undistort_full_; }
  BASIC::CloudPtr GetDSUndistort() const { return ds_undistort_; }
  BASIC::CloudPtr GetGlobalMap(bool use_voxel = true, float res = 0.1f) const {
    if (use_voxel && ds_world_ && !ds_world_->empty()) {
      return ds_world_;
    } else if (world_pc_ && !world_pc_->empty()) {
      return world_pc_;
    } else {
      return point_map_;
    }
  }
  bool IsInited() const { return kf_ && kf_->init_; }

protected:
  void stateWaitKFInit();
  void stateWaitMapInit();
  void stateProcess();
  virtual bool kf_init();
  virtual bool map_init();
  void Propagation_Undistort();
  void DownSample();
  void Observe();
  virtual void UpdateMap();
  virtual void Output();
  void caceData();
  void ProcessCaceMap();
  bool DetectZUPT();
  using StateFn = void (SuperLIO::*)();
  using OctVoxMapType = OctVoxMap<BASIC::V3, BASIC::scalar>;
  using KNNHeapType = KNNHeap<5, BASIC::V3>;
  StateFn state_fn_;
  ESKF::Ptr kf_;
  OctVoxMapType::Ptr ivox_;
  VoxelGridClosest<BASIC::PointType> voxel_grid_fliter_;
  std::shared_ptr<ROSWrapper> data_wrapper_;
  MeasureGroup measures_;
  
  bool flg_init_ = false;
  bool flg_first_scan_ = true;
  std::vector<DynamicState> propagate_states_;
  BASIC::CloudPtr scan_undistort_full_;
  BASIC::CloudPtr ds_undistort_;
  BASIC::CloudPtr point_map_, world_pc_, ds_world_;
  int frame_num_ = 0;
  BASIC::SE3 sys_init_pose_;
  BASIC::SE3 last_pose_;

  std::size_t effect_knn_num_ = 0;
  BASIC::VV3 points_world_v3_, points_body_v3_;
  alignas(64) bool effect_mask_[20000] = {false};
  alignas(64) bool effect_knn_mask_[20000] = {false};
  std::vector<int> effect_knn_idxs_;
  std::vector<std::pair<BASIC::M6, BASIC::V6>> H_R_;
  std::vector<std::array<double, 4>> abcd_vec_;
  int pcd_index_ = -1;

  Timer time_record_;
    // ZUPT (Zero Velocity Update) detection parameters
  static constexpr int ZUPT_MIN_DURATION = 5;
  
  // ZUPT detection variables
  bool is_zupt_active_ = false;
  int zupt_stationary_frames_ = 0;
  
  // Plane motion detection parameters
  static constexpr int PLANE_MOTION_WINDOW_SIZE = 1;
  static constexpr double PLANE_VELOCITY_THRESHOLD = 0.1;
  static constexpr double VERTICAL_VELOCITY_THRESHOLD = 0.15;
  
  // Plane motion detection variables
  std::deque<BASIC::V3> velocity_window_;
  bool is_plane_motion_ = false;
  double last_z_position_ = 0.0;
};

} // namespace LI2Sup

#endif
