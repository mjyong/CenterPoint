#include "velocity_calibration.h"
#include "config_helper.h"
#include "log.h"
#include <algorithm>
#include <yaml-cpp/yaml.h>

namespace planning {
namespace common {
namespace util {

namespace {
constexpr double kMaxCallGap = 0.5;
constexpr double kStandstillDesired = 1e-3;
} // namespace

VelocityCalibration::VelocityCalibration() {
  std::string calib_file =
      ConfigHelper::Instance()->GetConfDir() + "/velocity_calibration.yaml";

  YAML::Node node;
  try {
    node = YAML::LoadFile(calib_file);
  } catch (const YAML::Exception &e) {
    AWARN << "Failed to load velocity calibration file: " << calib_file
          << ", error: " << e.what();
    return;
  }

  const auto &integ = node["integral"];
  if (integ) {
    if (integ["leak_rate"]) {
      leak_rate_ = integ["leak_rate"].as<double>();
    }
    auto load_axis = [&](const std::string &key, IntegralParams &p) {
      const auto &axis = integ[key];
      if (!axis) {
        return;
      }
      if (axis["enable"]) {
        p.enabled = axis["enable"].as<bool>();
      }
      if (axis["ki"]) {
        p.ki = axis["ki"].as<double>();
      }
      if (axis["i_max"]) {
        p.i_max = axis["i_max"].as<double>();
      }
      if (axis["error_deadband"]) {
        p.error_deadband = axis["error_deadband"].as<double>();
      }
    };
    load_axis("vx", integral_params_[0]);
    load_axis("vy", integral_params_[1]);
    load_axis("omega", integral_params_[2]);

    for (const auto &p : integral_params_) {
      if (p.enabled && p.ki > 0.0 && p.i_max > 0.0) {
        AINFO << "Velocity integral compensation enabled: ki=" << p.ki
              << ", i_max=" << p.i_max
              << ", error_deadband=" << p.error_deadband
              << ", leak_rate=" << leak_rate_;
        break;
      }
    }
  }

  auto load_from_yaml = [&](const std::string &i_key, const std::string &o_key,
                            std::vector<Sample> &table) -> bool {
    if (!node[i_key] || !node[o_key]) {
      AWARN << "Missing key " << i_key << " or " << o_key
            << " in velocity calibration file";
      return false;
    }
    if (node[i_key].size() != node[o_key].size()) {
      AWARN << "Size mismatch for " << i_key << " and " << o_key;
      return false;
    }
    table.clear();
    for (size_t i = 0; i < node[i_key].size(); ++i) {
      Sample s;
      s.cmd = node[i_key][i].as<double>();
      s.real = node[o_key][i].as<double>();
      table.push_back(s);
    }
    if (table.size() < 2) {
      AWARN << "Table " << i_key << " has less than 2 samples";
      return false;
    }
    return true;
  };

  bool ok = true;
  ok &= load_from_yaml("i_vx", "o_vx", table_vx_);
  ok &= load_from_yaml("i_vy", "o_vy", table_vy_);
  ok &= load_from_yaml("i_omega", "o_omega", table_w_);

  if (!ok) {
    AWARN << "Velocity calibration table load failed";
    return;
  }

  auto sort_by_real = [](std::vector<Sample> &table) {
    std::sort(table.begin(), table.end(),
              [](const Sample &a, const Sample &b) { return a.real < b.real; });
  };
  sort_by_real(table_vx_);
  sort_by_real(table_vy_);
  sort_by_real(table_w_);

  loaded_ = true;
  AINFO << "Velocity calibration loaded from " << calib_file;
}

void VelocityCalibration::Reset() {
  integral_state_[0] = 0.0;
  integral_state_[1] = 0.0;
  integral_state_[2] = 0.0;
}

Eigen::Vector3d
VelocityCalibration::ComputeCommand(const Eigen::Vector3d &desired,
                                    const Eigen::Vector3d &actual) {
  double dt = 0.0;
  const auto now = std::chrono::steady_clock::now();
  if (has_last_call_time_) {
    const double gap =
        std::chrono::duration<double>(now - last_call_time_).count();
    if (gap > kMaxCallGap) {
      Reset();
    } else if (gap > 0.0) {
      dt = gap;
    }
  }
  last_call_time_ = now;
  has_last_call_time_ = true;

  const std::vector<Sample> *tables[3] = {&table_vx_, &table_vy_, &table_w_};
  const double desired_arr[3] = {desired.x(), desired.y(), desired.z()};
  const double actual_arr[3] = {actual.x(), actual.y(), actual.z()};

  Eigen::Vector3d cmd;
  double cmd_arr[3];
  for (int i = 0; i < 3; ++i) {
    double ff =
        loaded_ ? InterpolateInverse(*tables[i], desired_arr[i]) : desired_arr[i];

    if (dt > 0.0 && integral_params_[i].enabled &&
        integral_params_[i].ki > 0.0 && integral_params_[i].i_max > 0.0) {
      cmd_arr[i] =
          ff + ComputeIntegral(i, desired_arr[i], actual_arr[i], dt, ff,
                               loaded_ ? tables[i] : nullptr);
    } else {
      cmd_arr[i] = ff;
    }
  }
  cmd.x() = cmd_arr[0];
  cmd.y() = cmd_arr[1];
  cmd.z() = cmd_arr[2];
  return cmd;
}

double VelocityCalibration::ComputeIntegral(int idx, double desired,
                                            double actual, double dt,
                                            double ff_cmd,
                                            const std::vector<Sample> *table) {
  const IntegralParams &p = integral_params_[idx];
  double &integral = integral_state_[idx];

  if (std::fabs(desired) < kStandstillDesired) {
    integral = 0.0;
    return 0.0;
  }

  const double error = desired - actual;
  if (std::fabs(error) > p.error_deadband) {
    integral += error * dt;
  }

  integral *= std::max(0.0, 1.0 - leak_rate_ * dt);
  integral = std::clamp(integral, -p.i_max, p.i_max);

  double corr = p.ki * integral;
  if (table && table->size() >= 2) {
    const double lo = table->front().cmd;
    const double hi = table->back().cmd;
    if (lo < hi) {
      const double out = ff_cmd + corr;
      if (out > hi) {
        integral = std::min(integral, (hi - ff_cmd) / p.ki);
        corr = p.ki * integral;
      } else if (out < lo) {
        integral = std::max(integral, (lo - ff_cmd) / p.ki);
        corr = p.ki * integral;
      }
    }
  }
  return corr;
}

double
VelocityCalibration::InterpolateInverse(const std::vector<Sample> &table,
                                        double desired) const {
  if (table.size() < 2) {
    return desired;
  }

  for (size_t i = 0; i < table.size() - 1; ++i) {
    if (table[i].real <= desired && desired <= table[i + 1].real) {
      double t = (desired - table[i].real) /
                 (table[i + 1].real - table[i].real + 1e-6);
      return table[i].cmd + t * (table[i + 1].cmd - table[i].cmd);
    }
  }

  if (desired < table.front().real) {
    return table.front().cmd;
  }
  return table.back().cmd;
}

} // namespace util
} // namespace common
} // namespace planning
