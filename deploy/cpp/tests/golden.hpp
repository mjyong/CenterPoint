// Helpers of the parity tests: the golden data of tools/deploy/dump_golden.py and tolerant comparisons.
#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "dogperc/detections.hpp"
#include "dogperc/geometry.hpp"
#include "dogperc/npz.hpp"

namespace dogperc::test {

inline std::string golden(const std::string& name) { return std::string(DOGPERC_GOLDEN_DIR) + "/" + name; }

#define REQUIRE_GOLDEN()                                                                                     \
  if (!std::filesystem::exists(::dogperc::test::golden("summary.json")))                                    \
  GTEST_SKIP() << "no golden data in " DOGPERC_GOLDEN_DIR " (python3 tools/deploy/dump_golden.py --out ...)"

inline Npz load(const std::string& name) { return read_npz(golden(name)); }

inline Mat3 mat3(const NpyArray& a, int64_t i) {
  Mat3 R;
  for (int k = 0; k < 9; ++k) R(k / 3, k % 3) = a.at<double>(i * 9 + k);
  return R;
}
inline Mat4 mat4(const NpyArray& a, int64_t i) {
  Mat4 T;
  for (int k = 0; k < 16; ++k) T(k / 4, k % 4) = a.at<double>(i * 16 + k);
  return T;
}
inline Vec3 vec3(const NpyArray& a, int64_t i) { return Vec3(a.at<double>(3 * i), a.at<double>(3 * i + 1), a.at<double>(3 * i + 2)); }

// Detections of frame f from <prefix>_offsets / _boxes / ... (prefix "": offsets / boxes / ..., dets.npz)
inline Detections detections(const Npz& z, const std::string& prefix, int64_t f, const std::string& frame = "det") {
  const std::string p = prefix.empty() ? "" : prefix + "_";
  const NpyArray& off = member(z, p + "offsets");
  const int64_t a = off.at<int64_t>(f), b = off.at<int64_t>(f + 1);
  const NpyArray &bx = member(z, p + "boxes"), &v = member(z, p + "velocities");
  const NpyArray &s = member(z, p + "scores"), &l = member(z, p + "labels");
  const NpyArray& sc = member(z, p + "subclasses");
  Detections d;
  d.frame = frame;
  for (int64_t i = a; i < b; ++i) {
    Box box;
    for (int k = 0; k < 7; ++k) box[k] = bx.at<double>(7 * i + k);
    d.push(box, {v.at<double>(2 * i), v.at<double>(2 * i + 1)}, s.at<double>(i), l.at<int>(i), sc.at<int>(i));
  }
  return d;
}

inline double max_abs(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// same boxes in the same order, values within tol
inline void expect_same_detections(const Detections& a, const Detections& b, double tol, const std::string& what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a.labels[i], b.labels[i]) << what << " box " << i;
    EXPECT_EQ(a.subclasses[i], b.subclasses[i]) << what << " box " << i;
    EXPECT_NEAR(a.scores[i], b.scores[i], tol) << what << " box " << i;
    for (int k = 0; k < 7; ++k) {
      const double d = k == 6 ? wrap_angle(a.boxes[i][k] - b.boxes[i][k]) : a.boxes[i][k] - b.boxes[i][k];
      EXPECT_NEAR(d, 0.0, tol) << what << " box " << i << " field " << k;
    }
    for (int k = 0; k < 2; ++k) EXPECT_NEAR(a.velocities[i][k], b.velocities[i][k], tol) << what << " box " << i;
  }
}

}  // namespace dogperc::test
