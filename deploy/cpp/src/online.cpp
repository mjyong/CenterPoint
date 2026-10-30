#include "dogperc/online.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace dogperc {

namespace {

YAML::Node need(const YAML::Node& n, const std::string& key) {
  YAML::Node v = n[key];
  if (!v) throw Error("config: missing node." + key + " (write it with tools/deploy/export_cpp_config.py --robot)");
  return v;
}

void one_of(const std::string& key, const std::string& v, std::initializer_list<const char*> allowed) {
  for (const char* a : allowed)
    if (v == a) return;
  std::string list;
  for (const char* a : allowed) list += std::string(list.empty() ? "" : ", ") + a;
  throw Error("config: node." + key + " must be one of " + list + ", got " + v);
}

NodeConfig from_node(const YAML::Node& root) {
  const YAML::Node n = root["node"];
  if (!n) throw Error("config: no node: section (export_cpp_config.py --robot writes one)");
  NodeConfig c;
  c.lidar_topic = need(n, "lidar_topic").as<std::string>();
  c.time_field = need(n, "time_field").as<std::string>();
  c.time_mode = need(n, "time_mode").as<std::string>();
  c.noise_tag_mask = need(n, "noise_tag_mask").as<int>();
  c.lidar_time_offset = need(n, "lidar_time_offset").as<double>();
  c.imu_topic = need(n, "imu_topic").as<std::string>();
  c.imu_convention = need(n, "imu_convention").as<std::string>();
  c.localization = need(n, "localization").as<std::string>();
  c.odom_topic = need(n, "odom_topic").as<std::string>();
  c.odom_twist = need(n, "odom_twist").as<std::string>();
  c.inspvax_topic = need(n, "inspvax_topic").as<std::string>();
  const auto L = need(n, "ins_lever_arm").as<std::vector<double>>();
  if (L.size() != 3 || !std::all_of(L.begin(), L.end(), [](double v) { return std::isfinite(v); }))
    throw Error("config: node.ins_lever_arm needs 3 finite values (export_cpp_config.py resolves the default)");
  c.ins_lever_arm = Vec3(L[0], L[1], L[2]);
  c.ins_azimuth_offset_deg = need(n, "ins_azimuth_offset_deg").as<double>();
  c.world_frame = need(n, "world_frame").as<std::string>();
  c.publish_cloud = need(n, "publish_cloud").as<bool>();
  one_of("time_mode", c.time_mode, {"absolute", "absolute_ns", "relative", "relative_ns"});
  one_of("imu_convention", c.imu_convention, {"ros", "kygi"});
  one_of("localization", c.localization, {"odom", "inspvax"});
  one_of("odom_twist", c.odom_twist, {"none", "world", "body"});
  return c;
}

size_t type_size(uint8_t t) {
  switch (t) {
    case kInt8:
    case kUint8:
      return 1;
    case kInt16:
    case kUint16:
      return 2;
    case kInt32:
    case kUint32:
    case kFloat32:
      return 4;
    case kFloat64:
      return 8;
    default:
      return 0;                               // left out, as ros_utils.cloud_to_array does
  }
}

template <class T>
double as_double(const uint8_t* b) {
  T v;
  std::memcpy(&v, b, sizeof(T));
  return static_cast<double>(v);
}

// one field of every point, converted to float64 (exact for every PointField type)
struct FieldReader {
  uint32_t offset;
  uint8_t type;
  bool swap;
  double operator()(const uint8_t* point) const {
    uint8_t b[8];
    const size_t n = type_size(type);
    std::memcpy(b, point + offset, n);          // fields need not be aligned (XT32: float64 at offset 18)
    if (swap) std::reverse(b, b + n);
    switch (type) {
      case kInt8: return as_double<int8_t>(b);
      case kUint8: return as_double<uint8_t>(b);
      case kInt16: return as_double<int16_t>(b);
      case kUint16: return as_double<uint16_t>(b);
      case kInt32: return as_double<int32_t>(b);
      case kUint32: return as_double<uint32_t>(b);
      case kFloat32: return as_double<float>(b);
      default: return as_double<double>(b);
    }
  }
};

std::optional<FieldReader> field(const CloudView& c, const std::string& name) {
  for (const auto& f : c.fields)
    if (f.name == name && type_size(f.datatype)) {
      if (f.offset + type_size(f.datatype) > c.point_step)
        throw Error("PointCloud2: field " + name + " ends past point_step");
      return FieldReader{f.offset, f.datatype, c.is_bigendian};   // the hosts are little-endian
    }
  return std::nullopt;
}

}  // namespace

NodeConfig parse_node_config(const std::string& text) {
  try {
    return from_node(YAML::Load(text));
  } catch (const YAML::Exception& e) {
    throw Error(std::string("config: ") + e.what());
  }
}

NodeConfig load_node_config(const std::string& path) {
  try {
    return from_node(YAML::LoadFile(path));
  } catch (const YAML::Exception& e) {
    throw Error(path + ": " + e.what());
  }
}

LidarScan scan_from_cloud(const CloudView& c, const std::string& time_field, const std::string& time_mode,
                          int noise_tag_mask, const std::string& intensity_field) {
  const size_t n = static_cast<size_t>(c.width) * c.height;
  if (n && c.size < n * c.point_step) throw Error("PointCloud2: fewer bytes than width * height * point_step");
  const auto fx = field(c, "x"), fy = field(c, "y"), fz = field(c, "z");
  if (!fx || !fy || !fz) throw Error("PointCloud2: no x / y / z fields");
  const auto fi = field(c, intensity_field), ft = field(c, time_field);
  const auto ftag = noise_tag_mask ? field(c, "tag") : std::nullopt;
  std::vector<float> xyz, inten;
  std::vector<double> times;
  xyz.reserve(3 * n);
  inten.reserve(n);
  if (ft) times.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* pt = c.data + i * c.point_step;
    const float x = static_cast<float>((*fx)(pt)), y = static_cast<float>((*fy)(pt)), z = static_cast<float>((*fz)(pt));
    // np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0): no return at the origin
    if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) continue;
    if (!((std::abs(x) + std::abs(y)) + std::abs(z) > 0.0f)) continue;
    if (ftag) {
      const double tag = (*ftag)(pt);
      const int64_t bits = std::isfinite(tag) ? static_cast<int64_t>(tag) : std::numeric_limits<int64_t>::min();
      if (bits & noise_tag_mask) continue;
    }
    xyz.insert(xyz.end(), {x, y, z});
    inten.push_back(fi ? static_cast<float>((*fi)(pt)) : 0.0f);
    if (ft) {
      double t = (*ft)(pt);
      if (time_mode == "absolute_ns") t = t * 1e-9;
      else if (time_mode == "relative") t = c.stamp + t;
      else if (time_mode == "relative_ns") t = c.stamp + t * 1e-9;
      times.push_back(t);
    }
  }
  if (ft) return LidarScan::make(std::move(xyz), std::move(inten), std::move(times));
  return LidarScan::make(std::move(xyz), std::move(inten), {}, c.stamp);
}

OnlineInputs::OnlineInputs(NodeConfig cfg, PerceptionPipeline& pipe, LogFn log)
    : cfg_(std::move(cfg)), pipe_(pipe), log_(std::move(log)) {
  one_of("localization", cfg_.localization, {"odom", "inspvax"});
  one_of("time_mode", cfg_.time_mode, {"absolute", "absolute_ns", "relative", "relative_ns"});
}

void OnlineInputs::log(LogLevel level, const std::string& key, double every_s, const std::string& text) {
  if (!log_) return;
  const auto now = std::chrono::steady_clock::now();
  auto it = last_log_.find(key);
  if (every_s > 0 && it != last_log_.end() && std::chrono::duration<double>(now - it->second).count() < every_s) return;
  last_log_[key] = now;
  log_(level, text);
}

void OnlineInputs::reset() {
  pipe_.reset();
  origin_.reset();
  heading_ = HeadingCheck();
  heading_warned_ = false;
  last_stamp_.clear();
  restart_.reset();
  before_.clear();
}

bool OnlineInputs::stamp(const char* kind, double t) {
  char buf[160];
  auto it = last_stamp_.find(kind);
  if (it != last_stamp_.end() && t < it->second - kTimeJumpBack) {
    std::snprintf(buf, sizeof(buf), "%s stamps jumped back %.1f s (bag played again or clock reset): starting over",
                  kind, it->second - t);
    log(LogLevel::Warn, "jump", 0.0, buf);
    std::map<std::string, double> before = std::move(last_stamp_);
    reset();
    restart_ = t;
    before_ = std::move(before);
    it = last_stamp_.end();
  } else if (it == last_stamp_.end() && restart_) {
    const auto b = before_.find(kind);
    if (t > *restart_ + kTimeJumpBack && b != before_.end() && t > b->second - kTimeJumpBack) {
      std::snprintf(buf, sizeof(buf), "%s message from before the restart dropped", kind);
      log(LogLevel::Warn, "before restart", 2.0, buf);
      return false;
    }
  }
  if (it == last_stamp_.end()) last_stamp_[kind] = t;
  else it->second = std::max(it->second, t);
  return true;
}

void OnlineInputs::on_imu(double t, const Vec3& w) {
  if (!stamp("imu", t)) return;
  pipe_.on_imu(t, cfg_.imu_convention == "kygi" ? gyro_from_kygi_imu(w) : w);
}

void OnlineInputs::on_odom(const OdomMsg& m) {
  if (!stamp("odom", m.t)) return;
  const Mat3 R = quat_to_R(m.q_xyzw[0], m.q_xyzw[1], m.q_xyzw[2], m.q_xyzw[3]);
  std::optional<Vec3> v;
  if (cfg_.odom_twist != "none") v = cfg_.odom_twist == "body" ? Vec3(R * m.v_linear) : m.v_linear;
  pipe_.on_odometry(m.t, R, m.p, v);
}

void OnlineInputs::on_inspvax(const InsSample& s) {
  if (!stamp("inspvax", s.stamp)) return;
  char buf[256];
  if (!origin_) {
    if (s.ins_status != kInsSolutionGood) {
      log(LogLevel::Warn, "wait", 2.0, "waiting for a good INS solution");
      return;
    }
    origin_.emplace(s.latitude, s.longitude, s.height);
    std::snprintf(buf, sizeof(buf), "ENU origin %.8f %.8f %.2f", s.latitude, s.longitude, s.height);
    log(LogLevel::Info, "origin", 0.0, buf);
  }
  const auto got = pose_from_inspvax(s, *origin_, cfg_.ins_lever_arm, cfg_.ins_azimuth_offset_deg);
  if (!got) {
    std::snprintf(buf, sizeof(buf), "INS solution not good (ins_status %d)", s.ins_status);
    log(LogLevel::Warn, "not good", 2.0, buf);
    return;
  }
  if (s.pos_type != kInsRtkFixed) {
    std::snprintf(buf, sizeof(buf), "RTK not fixed (pos_type %d)", s.pos_type);
    log(LogLevel::Warn, "rtk", 5.0, buf);
  }
  heading_.add(got->R, got->v);
  if (heading_.wrong() && !heading_warned_) {
    heading_warned_ = true;
    std::snprintf(buf, sizeof(buf),
                  "the robot walks backwards in the body frame: ins_azimuth_offset_deg %.0f does not match the INS "
                  "antennas (180 swapped as on 0921, 0 fixed)", cfg_.ins_azimuth_offset_deg);
    log(LogLevel::Error, "heading", 0.0, buf);
  }
  pipe_.on_odometry(got->t, got->R, got->p, got->v);
}

LidarScan OnlineInputs::scan(const CloudView& cloud) const {
  return scan_from_cloud(cloud, cfg_.time_field, cfg_.time_mode, cfg_.noise_tag_mask).shifted(cfg_.lidar_time_offset);
}

std::optional<PipelineOutput> OnlineInputs::on_cloud(const CloudView& cloud, const Detections* detections) {
  return on_scan(scan(cloud), detections);
}

std::optional<PipelineOutput> OnlineInputs::on_scan(const LidarScan& s, const Detections* detections) {
  if (!stamp("cloud", s.stamp)) return std::nullopt;
  auto out = pipe_.on_scan(s, detections);
  if (!out) log(LogLevel::Warn, "no pose", 2.0, "no pose for scan yet (waiting for odometry / IMU)");
  return out;
}

std::vector<PlannerObject> planner_objects(const PerceptionPipeline& pipe, const PipelineOutput& out) {
  std::vector<LocalTrack> tracks;
  std::vector<Prediction> preds;
  pipe.to_local(out, tracks, preds);
  std::map<int, const Prediction*> by_id;
  for (const auto& p : preds) by_id[p.track_id] = &p;
  std::vector<PlannerObject> objects;
  for (const auto& t : tracks) {
    PlannerObject o;
    o.track = t;
    auto it = by_id.find(t.id);
    if (it != by_id.end()) {
      const Prediction& p = *it->second;
      o.times = p.times;
      std::vector<size_t> order(p.probs.size());
      for (size_t k = 0; k < order.size(); ++k) order[k] = k;
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p.probs[a] > p.probs[b]; });
      for (size_t k : order) o.paths.push_back({p.probs[k], p.modes[k], p.covs[k]});
    }
    objects.push_back(std::move(o));
  }
  return objects;
}

}  // namespace dogperc
