#include "dogperc/sequence.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>

#include "dogperc/npz.hpp"

namespace dogperc {

namespace fs = std::filesystem;

SequenceReader::SequenceReader(const std::string& dir, std::optional<double> offset) : dir_(dir) {
  const YAML::Node meta = YAML::LoadFile((fs::path(dir) / "meta.json").string());
  lidar_time_offset_ = offset ? *offset : (meta["lidar_time_offset"] ? meta["lidar_time_offset"].as<double>() : 0.0);
  const fs::path scans = fs::path(dir) / "scans";
  if (fs::exists(scans))
    for (const auto& e : fs::directory_iterator(scans))
      if (e.path().extension() == ".npz") scan_files_.push_back(e.path().string());
  std::sort(scan_files_.begin(), scan_files_.end());
  const fs::path imu = fs::path(dir) / "imu.npy", odom = fs::path(dir) / "odom.npy";
  if (fs::exists(imu)) imu_ = read_npy(imu.string()).to<double>();
  if (fs::exists(odom)) odom_ = read_npy(odom.string()).to<double>();
}

LidarScan SequenceReader::load_scan(size_t index) const {
  const Npz z = read_npz(scan_files_.at(index));
  LidarScan s;
  s.xyz = member(z, "xyz").to<float>();
  s.intensity = member(z, "intensity").to<float>();
  s.point_times = member(z, "t").to<double>();
  s.stamp = member(z, "stamp").scalar();
  return s.shifted(lidar_time_offset_);
}

std::vector<SequenceEvent> SequenceReader::events(double odom_latency) const {
  struct Item {
    double t;
    int order;
    size_t seq;
    SequenceEvent ev;
  };
  std::vector<Item> items;
  std::vector<double> stamps;
  for (size_t i = 0; i < scan_files_.size(); ++i) {
    const Npz z = read_npz(scan_files_[i]);
    stamps.push_back(member(z, "stamp").scalar() + lidar_time_offset_);
  }
  size_t seq = 0;
  for (size_t r = 0; r + 4 <= imu_.size(); r += 4) {
    const double* row = &imu_[r];
    items.push_back({row[0], 0, seq++, ImuEvent{row[0], Vec3(row[1], row[2], row[3])}});
  }
  for (size_t r = 0; r + 11 <= odom_.size(); r += 11) {
    const double* row = &odom_[r];
    std::optional<Vec3> v;
    if (!(std::isnan(row[8]) || std::isnan(row[9]) || std::isnan(row[10]))) v = Vec3(row[8], row[9], row[10]);
    items.push_back({row[0] + odom_latency, 1, seq++,
                     OdomEvent{row[0], quat_to_R(row[4], row[5], row[6], row[7]), Vec3(row[1], row[2], row[3]), v}});
  }
  for (size_t i = 0; i < stamps.size(); ++i) items.push_back({stamps[i], 2, seq++, ScanEvent{i, stamps[i]}});
  // Python sorts the (t, order, event) list by (t, order), stably
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) { return a.t < b.t || (a.t == b.t && a.order < b.order); });
  const double t_lo = stamps.empty() ? -kInf : stamps.front() - 1.0;
  const double t_hi = stamps.empty() ? kInf : stamps.back();
  std::vector<SequenceEvent> out;
  for (const auto& it : items) {
    if (it.t < t_lo || it.t > t_hi + 1e-6) continue;
    out.push_back(it.ev);
  }
  return out;
}

}  // namespace dogperc
