// Classes, per-class values and small helpers shared by every stage (lidar_3d_detection/detection/boxes.py).
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lidar3d {

// Everything after the detector (tracker, predictor, ROS) sees these three classes; the vehicle subclass
// rides along (Detections::subclasses).
constexpr int kNumClasses = 3;
extern const std::array<const char*, kNumClasses> kClasses;          // vehicle, pedestrian, cyclist
constexpr int kNumSubclasses = 5;
extern const std::array<const char*, kNumSubclasses> kSubclasses;    // car, truck, bus, pedestrian, cyclist

int class_index(const std::string& name);      // -1 when unknown
int subclass_index(const std::string& name);
int default_subclass(int label);               // pedestrian / cyclist: their own subclass, vehicle: -1

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A value per class: the Python configs hold either a float or a {class: value} dict where a missing
// class reads 0 (``score_threshold_array`` / ``_per_class``).
struct PerClass {
  std::array<double, kNumClasses> v{0.0, 0.0, 0.0};
  PerClass() = default;
  explicit PerClass(double all) { v.fill(all); }
  PerClass(double vehicle, double pedestrian, double cyclist) : v{vehicle, pedestrian, cyclist} {}
  double operator[](int label) const { return label >= 0 && label < kNumClasses ? v[label] : 0.0; }
  double& operator[](int label) { return v.at(label); }
};

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace lidar3d
