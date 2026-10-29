#include "dogperc/localization.hpp"

#include <cmath>

namespace dogperc {

namespace {

constexpr double kA = 6378137.0;
constexpr double kE2 = 6.69437999014e-3;

Vec3 lla_to_ecef(double lat, double lon, double h) {
  lat = lat * (kPi / 180.0);    // np.radians: x * (pi / 180)
  lon = lon * (kPi / 180.0);
  const double sl = std::sin(lat), cl = std::cos(lat), sn = std::sin(lon), cn = std::cos(lon);
  const double n = kA / std::sqrt(1.0 - kE2 * sl * sl);
  return Vec3((n + h) * cl * cn, (n + h) * cl * sn, (n * (1.0 - kE2) + h) * sl);
}

}  // namespace

EnuOrigin::EnuOrigin(double lat, double lon, double h) : ecef_(lla_to_ecef(lat, lon, h)) {
  const double lat0 = lat * (kPi / 180.0), lon0 = lon * (kPi / 180.0);
  const double sl = std::sin(lat0), cl = std::cos(lat0), sn = std::sin(lon0), cn = std::cos(lon0);
  R_ << -sn, cn, 0.0, -sl * cn, -sl * sn, cl, cl * cn, cl * sn, sl;
}

Vec3 EnuOrigin::enu(double lat, double lon, double h) const { return R_ * (lla_to_ecef(lat, lon, h) - ecef_); }

std::optional<InsPose> pose_from_inspvax(const InsSample& s, const EnuOrigin& origin, const Vec3& lever_arm,
                                         double azimuth_offset_deg) {
  if (s.ins_status != kInsSolutionGood) return std::nullopt;
  InsPose out;
  out.t = s.stamp;
  out.R = rot_z((azimuth_offset_deg - s.azimuth) * kPi / 180.0);
  out.p = origin.enu(s.latitude, s.longitude, s.height) - out.R * lever_arm;
  out.v = Vec3(s.east_velocity, s.north_velocity, s.up_velocity);
  return out;
}

void HeadingCheck::add(const Mat3& R, const Vec3& v) {
  if (!(std::isfinite(v[0]) && std::isfinite(v[1])) || std::hypot(v[0], v[1]) <= min_speed_) return;
  if (R(0, 1) * v[0] + R(1, 1) * v[1] > 0) ++n_fwd_;
  else ++n_back_;
}

}  // namespace dogperc
