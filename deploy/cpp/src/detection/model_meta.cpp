#include "dogperc/detection/model_meta.hpp"

#include <yaml-cpp/yaml.h>

namespace dogperc {

namespace {

constexpr int kMetaVersion = 1;

YAML::Node need(const YAML::Node& n, const std::string& key) {
  YAML::Node v = n[key];
  if (!v) throw Error("model_meta.json: missing " + key);
  return v;
}

template <class T, size_t N>
std::array<T, N> arr(const YAML::Node& n, const std::string& key) {
  const auto v = need(n, key).as<std::vector<T>>();
  if (v.size() < N) throw Error("model_meta.json: " + key + " needs " + std::to_string(N) + " values");
  std::array<T, N> a{};
  for (size_t i = 0; i < N; ++i) a[i] = v[i];
  return a;
}

}  // namespace

std::vector<int> ModelMeta::num_classes() const {
  std::vector<int> n;
  for (const auto& t : task_classes) n.push_back(static_cast<int>(t.size()));
  return n;
}

int ModelMeta::head_channels(int task, const std::string& key) const {
  if (key == "reg" || key == "rot" || key == "vel") return 2;
  if (key == "height") return 1;
  if (key == "dim") return 3;
  if (key == "hm") return static_cast<int>(task_classes.at(task).size());
  throw Error("model_meta: unknown head " + key);
}

ModelMeta load_model_meta(const std::string& path) {
  YAML::Node d;
  try {
    d = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw Error(path + ": " + e.what());
  }
  ModelMeta m;
  m.version = d["version"] ? d["version"].as<int>() : 0;
  if (m.version != kMetaVersion)
    throw Error(path + ": model_meta version " + std::to_string(m.version) + ", this code reads " +
                std::to_string(kMetaVersion) + " (re-export the ONNX)");
  m.voxel_size = arr<double, 3>(d, "voxel_size");
  m.pc_range = arr<double, 6>(d, "pc_range");
  m.grid_size = arr<int, 3>(d, "grid_size");
  m.max_points = need(d, "max_points").as<int>();
  m.max_voxels = need(d, "max_voxels").as<int>();
  m.max_pillars = need(d, "max_pillars").as<int>();
  m.num_point_features = need(d, "num_point_features").as<int>();
  m.pfn_channels = need(d, "pfn_channels").as<int>();
  m.bev_channels = need(d, "bev_channels").as<int>();
  m.z_range = arr<double, 2>(d, "z_range");
  const YAML::Node fr = need(d, "frame");
  m.frame.nusc_axes = need(fr, "nusc_axes").as<bool>();
  m.frame.z_offset = need(fr, "z_offset").as<double>();
  for (const auto& t : need(d, "tasks")) m.task_classes.push_back(need(t, "class_names").as<std::vector<std::string>>());
  m.head_keys = need(d, "head_keys").as<std::vector<std::string>>();
  const YAML::Node tc = need(d, "test_cfg");
  m.decoder.pc_range = arr<double, 2>(tc, "pc_range");
  m.decoder.voxel_size = arr<double, 2>(tc, "voxel_size");
  m.decoder.out_size_factor = need(tc, "out_size_factor").as<int>();
  m.decoder.score_threshold = need(tc, "score_threshold").as<double>();
  m.decoder.post_center_limit_range = arr<double, 6>(tc, "post_center_limit_range");
  m.decoder.min_radius = need(tc, "min_radius").as<std::vector<double>>();
  m.decoder.nms_post_max_size = need(need(tc, "nms"), "nms_post_max_size").as<int>();
  m.decoder.pre_max_size = tc["pre_max_size"] ? tc["pre_max_size"].as<int>() : 1000;
  m.label_map = need(d, "label_map").as<std::vector<int>>();
  m.subclass_map = need(d, "subclass_map").as<std::vector<int>>();
  const YAML::Node nms = need(d, "merged_nms");
  for (auto it = nms.begin(); it != nms.end(); ++it) {
    const int c = class_index(it->first.as<std::string>());
    if (c < 0 || it->second.IsNull()) continue;
    MergedNms r;
    r.set = true;
    if (it->second.IsSequence()) {
      const auto v = it->second.as<std::vector<double>>();
      r.radius = v.at(0);
      if (v.size() >= 3) {
        r.weak_radius = v[1];
        r.weak_ratio = v[2];
      }
    } else {
      r.radius = it->second.as<double>();
    }
    m.merged_nms[c] = r;
  }
  const YAML::Node onnx = need(d, "onnx");
  const YAML::Node pfn = need(onnx, "pfn"), rpn = need(onnx, "rpn");
  m.pfn_input = need(pfn, "input").as<std::string>();
  m.pfn_output = need(pfn, "output").as<std::string>();
  m.pfn_dynamic = pfn["dynamic"] ? pfn["dynamic"].as<bool>() : false;
  m.rpn_input = need(rpn, "input").as<std::string>();
  m.rpn_outputs = need(rpn, "outputs").as<std::vector<std::string>>();
  if (static_cast<int>(m.decoder.min_radius.size()) != m.num_tasks())
    throw Error(path + ": test_cfg.min_radius needs one value per task");
  if (m.rpn_outputs.size() != m.task_classes.size() * m.head_keys.size())
    throw Error(path + ": onnx.rpn.outputs does not match tasks x head_keys");
  return m;
}

}  // namespace dogperc
