// model_meta.json: what a deployed Pillar model needs besides its two engines (detection/model_meta.py).
// Written by tools/dog/export_pillar_onnx.py; a retrained model needs no code change.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "dogperc/detections.hpp"

namespace dogperc {

struct DecoderConfig {                           // test_cfg of decode.decode_centerpoint
  std::array<double, 2> pc_range{};              // x, y min of the BEV grid
  std::array<double, 2> voxel_size{};
  int out_size_factor = 4;
  double score_threshold = 0.1;
  std::array<double, 6> post_center_limit_range{};
  std::vector<double> min_radius;                // per task, squared meters (det3d convention)
  int nms_post_max_size = 83;
  int pre_max_size = 1000;
};

struct ModelMeta {
  std::array<double, 3> voxel_size{};
  std::array<double, 6> pc_range{};
  std::array<int, 3> grid_size{};                // nx, ny, nz
  int max_points = 20;                           // per pillar
  int max_voxels = 30000;                        // det3d's limit
  int max_pillars = 30000;                       // P_max of the PFN engine
  int num_point_features = 5;                    // x, y, z, intensity, dt
  int pfn_channels = 10;                         // decorated features
  int bev_channels = 64;
  std::array<double, 2> z_range{};               // det-frame crop
  ModelFrame frame;
  std::vector<std::vector<std::string>> task_classes;
  std::vector<std::string> head_keys;            // HEAD_ORDER subset: reg, height, dim, rot, vel, hm
  DecoderConfig decoder;
  std::vector<int> label_map;                    // model class -> dog class (-1 dropped)
  std::vector<int> subclass_map;                 // model class -> subclass (-1 none)
  MergedNmsMap merged_nms;
  std::string pfn_input = "pillars", pfn_output = "pillar_features", rpn_input = "bev";
  bool pfn_dynamic = false;                      // free pillar dimension (export --dynamic-pillars)
  std::vector<std::string> rpn_outputs;          // task<t>_<key>
  int version = 1;

  int num_tasks() const { return static_cast<int>(task_classes.size()); }
  std::vector<int> num_classes() const;
  // channels of head `key` for task t
  int head_channels(int task, const std::string& key) const;
  int feature_x() const { return grid_size[0] / decoder.out_size_factor; }
  int feature_y() const { return grid_size[1] / decoder.out_size_factor; }
};

ModelMeta load_model_meta(const std::string& path);

}  // namespace dogperc
