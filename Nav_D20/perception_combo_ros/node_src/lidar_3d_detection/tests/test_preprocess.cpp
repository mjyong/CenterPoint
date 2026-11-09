// Pose buffer interpolation / extrapolation and the per-sweep preprocessing against Python.
#include "lidar3d/preprocess.hpp"
#include "golden.hpp"

namespace lidar3d::test {

namespace {

PoseBuffer buffer_from(const Npz& z, const std::string& t, const std::string& R, const std::string& p) {
  PoseBuffer buf;
  const auto ts = member(z, t).to<double>();
  for (size_t i = 0; i < ts.size(); ++i) buf.add(ts[i], mat3(member(z, R), i), vec3(member(z, p), i));
  return buf;
}

}  // namespace

TEST(PoseBuffer, InterpolationAndExtrapolationMatchPython) {
  REQUIRE_GOLDEN();
  const Npz z = load("pose_buffer.npz");
  const PoseBuffer buf = buffer_from(z, "t", "R", "p");
  const auto q = member(z, "query").to<double>();
  std::vector<Mat3> R;
  std::vector<Vec3> p;
  buf.interpolate(q, R, p);
  for (size_t i = 0; i < q.size(); ++i) {
    EXPECT_LT((R[i] - mat3(member(z, "R_query"), i)).cwiseAbs().maxCoeff(), 1e-12) << i;
    EXPECT_LT((p[i] - vec3(member(z, "p_query"), i)).cwiseAbs().maxCoeff(), 1e-12) << i;
  }
  EXPECT_THROW(buf.pose_at(q.back() + 10.0), PoseOutOfRange);
  EXPECT_THROW(PoseBuffer().pose_at(0.0), PoseOutOfRange);
}

TEST(Preprocess, DedupSelfFilterNeighborsDeskewMatchPython) {
  REQUIRE_GOLDEN();
  const Npz z = load("preprocess.npz");
  LidarScan scan;
  scan.xyz = member(z, "xyz").to<float>();
  scan.intensity = member(z, "intensity").to<float>();
  scan.point_times = member(z, "t").to<double>();
  scan.stamp = member(z, "stamp").scalar();

  const LidarScan dd = dedup_returns(scan);
  const auto want_xyz = member(z, "dedup_xyz").to<float>();
  ASSERT_EQ(dd.xyz.size(), want_xyz.size());
  EXPECT_EQ(dd.xyz, want_xyz);
  EXPECT_EQ(dd.point_times, member(z, "dedup_t").to<double>());

  // the sim's T_body_lidar is the sequence's
  const PipelineConfig cfg = load_config(golden("config.yaml"));
  const SelfFilter sf(SelfFilterConfig{}, cfg.preprocess.T_body_lidar);
  const auto fin = member(z, "filter_xyz").to<float>();       // the finite points of the dedup output
  const auto keep = sf(fin.data(), fin.size() / 3);
  const auto want_keep = member(z, "keep").to<int>();
  ASSERT_EQ(keep.size(), want_keep.size());
  size_t diff = 0;
  for (size_t i = 0; i < keep.size(); ++i) diff += (keep[i] != 0) != (want_keep[i] != 0);
  EXPECT_EQ(diff, 0u);

  const auto near = member(z, "near_xyz").to<float>();
  const auto cnt = neighbor_counts(near.data(), near.size() / 3, 0.2);
  EXPECT_EQ(cnt, member(z, "near_counts").to<int64_t>());

  PreprocessConfig pc;
  pc.T_body_lidar = cfg.preprocess.T_body_lidar;
  pc.det_frame.base_height = 0.45;
  const PoseBuffer buf = buffer_from(z, "buf_t", "buf_R", "buf_p");
  const Preprocessor pre(pc, buf);
  std::vector<double> xyz_b;
  std::vector<float> inten;
  pre.sweep_to_body(scan, xyz_b, inten);
  const auto want_b = member(z, "body_xyz").to<double>();
  ASSERT_EQ(xyz_b.size(), want_b.size());
  EXPECT_LT(max_abs(xyz_b, want_b), 1e-9);
  EXPECT_EQ(inten, member(z, "body_intensity").to<float>());
}

}  // namespace lidar3d::test
