// .npz IO, geometry, configuration, INSPVAX and the assignment solver against Python.
#include <cstdio>
#include <filesystem>

#include "dogperc/config.hpp"
#include "dogperc/localization.hpp"
#include "dogperc/sequence.hpp"
#include "dogperc/tracking/association.hpp"
#include "golden.hpp"

namespace dogperc::test {

TEST(Npz, RoundTrip) {
  const std::string path = (std::filesystem::temp_directory_path() / "dogperc_roundtrip.npz").string();
  NpzWriter w;
  w.add<float>("f", {1.5f, -2.0f, 3.25f, 4.0f, 5.0f, 6.0f}, {2, 3});
  w.add<int64_t>("i", {7, -8, 9});
  w.add_scalar<double>("s", 0.125);
  w.write(path);
  const Npz z = read_npz(path);
  ASSERT_EQ(member(z, "f").shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(member(z, "f").to<float>()[5], 6.0f);
  EXPECT_EQ(member(z, "i").to<int64_t>()[1], -8);
  EXPECT_TRUE(member(z, "s").shape.empty());
  EXPECT_EQ(member(z, "s").scalar(), 0.125);
  std::remove(path.c_str());
}

TEST(Npz, CompressedSequenceScan) {
  REQUIRE_GOLDEN();
  const SequenceReader r(golden("seq"));
  ASSERT_GT(r.num_scans(), 0u);
  const LidarScan s = r.load_scan(0);
  EXPECT_GT(s.size(), 1000u);
  EXPECT_EQ(s.xyz.size(), 3 * s.size());
  EXPECT_EQ(s.point_times.size(), s.size());
}

TEST(Geometry, MatchesScipyAndNumpy) {
  REQUIRE_GOLDEN();
  const Npz g = load("geometry.npz");
  const NpyArray& rv = member(g, "rotvec");
  for (int64_t i = 0; i < rv.dim(0); ++i) {
    const Mat3 R = so3_exp(vec3(rv, i));
    EXPECT_LT((R - mat3(member(g, "so3_exp"), i)).cwiseAbs().maxCoeff(), 1e-14) << i;
    EXPECT_LT((so3_log(R) - vec3(member(g, "so3_log"), i)).cwiseAbs().maxCoeff(), 1e-12) << i;
  }
  const NpyArray& q = member(g, "quat");
  for (int64_t i = 0; i < q.dim(0); ++i) {
    const Mat3 R = quat_to_R(q.at<double>(4 * i), q.at<double>(4 * i + 1), q.at<double>(4 * i + 2), q.at<double>(4 * i + 3));
    EXPECT_LT((R - mat3(member(g, "quat_R"), i)).cwiseAbs().maxCoeff(), 1e-14) << i;
  }
  const NpyArray& e = member(g, "euler");
  for (int64_t i = 0; i < e.dim(0); ++i) {
    const Mat3 R = rot_zyx(e.at<double>(3 * i), e.at<double>(3 * i + 1), e.at<double>(3 * i + 2));
    EXPECT_LT((R - mat3(member(g, "rot_zyx"), i)).cwiseAbs().maxCoeff(), 1e-14) << i;
  }
  const auto a = member(g, "angles").to<double>(), w = member(g, "wrapped").to<double>();
  for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(wrap_angle(a[i]), w[i]) << a[i];
}

void expect_same(const PerClass& a, const PerClass& b, const std::string& what) {
  for (int c = 0; c < kNumClasses; ++c) EXPECT_EQ(a[c], b[c]) << what << " " << kClasses[c];
}

TEST(Config, CppDefaultsEqualThePythonDataclasses) {
  REQUIRE_GOLDEN();
  const PipelineConfig py = load_config(golden("default.yaml"));
  const PipelineConfig cc;
  const auto &a = cc.preprocess, &b = py.preprocess;
  EXPECT_EQ((a.T_body_lidar - b.T_body_lidar).cwiseAbs().maxCoeff(), 0.0);
  EXPECT_EQ(a.num_sweeps, b.num_sweeps);
  EXPECT_EQ(a.max_time_span, b.max_time_span);
  EXPECT_EQ(a.deskew, b.deskew);
  EXPECT_EQ(a.time_resolution, b.time_resolution);
  EXPECT_EQ(a.dedup_returns, b.dedup_returns);
  EXPECT_EQ(a.intensity_scale, b.intensity_scale);
  EXPECT_EQ(a.self_filter.body_boxes, b.self_filter.body_boxes);
  EXPECT_EQ(a.self_filter.min_range, b.self_filter.min_range);
  EXPECT_EQ(a.self_filter.max_range, b.self_filter.max_range);
  EXPECT_EQ(a.self_filter.outlier_radius, b.self_filter.outlier_radius);
  EXPECT_EQ(a.self_filter.outlier_min_neighbors, b.self_filter.outlier_min_neighbors);
  EXPECT_EQ(a.self_filter.outlier_max_range, b.self_filter.outlier_max_range);
  const auto &d = a.det_frame, &e = b.det_frame;
  EXPECT_EQ(d.base_height, e.base_height);
  EXPECT_EQ(d.auto_ground, e.auto_ground);
  EXPECT_EQ(d.auto_level, e.auto_level);
  EXPECT_EQ(d.ground_ring, e.ground_ring);
  EXPECT_EQ(d.ground_band, e.ground_band);
  EXPECT_EQ(d.ground_min_points, e.ground_min_points);
  EXPECT_EQ(d.ground_alpha, e.ground_alpha);
  EXPECT_EQ(d.ground_max_correction, e.ground_max_correction);
  EXPECT_EQ(d.ground_max_tilt, e.ground_max_tilt);
  EXPECT_EQ(cc.pose_buffer.max_age, py.pose_buffer.max_age);
  EXPECT_EQ(cc.pose_buffer.max_extrapolation, py.pose_buffer.max_extrapolation);
  EXPECT_EQ(cc.pose_buffer.extrapolation_window, py.pose_buffer.extrapolation_window);
  EXPECT_EQ(cc.imu.gyro_bias, py.imu.gyro_bias);
  EXPECT_EQ(cc.imu.imu_history, py.imu.imu_history);
  const auto &f = cc.detection_filter, &g = py.detection_filter;
  EXPECT_EQ(f.enabled, g.enabled);
  expect_same(f.min_points, g.min_points, "min_points");
  EXPECT_EQ(f.box_margin, g.box_margin);
  EXPECT_EQ(f.max_float, g.max_float);
  EXPECT_EQ(f.max_sink, g.max_sink);
  EXPECT_EQ(f.local_ground, g.local_ground);
  EXPECT_EQ(f.ground_cell, g.ground_cell);
  EXPECT_EQ(f.ground_ring, g.ground_ring);
  EXPECT_EQ(f.ground_percentile, g.ground_percentile);
  EXPECT_EQ(f.ground_min_cells, g.ground_min_cells);
  EXPECT_EQ(f.cross_class_nms, g.cross_class_nms);
  EXPECT_EQ(f.cross_class_margin, g.cross_class_margin);
  expect_same(cc.score_thresholds, py.score_thresholds, "score_thresholds");
  const auto &t = cc.tracker, &u = py.tracker;
  for (int c = 0; c < kNumClasses; ++c) {
    const auto &p = t.class_params[c], &q = u.class_params[c];
    EXPECT_EQ(p.gate, q.gate) << kClasses[c];
    EXPECT_EQ(p.max_age, q.max_age) << kClasses[c];
    EXPECT_EQ(p.min_hits, q.min_hits) << kClasses[c];
    EXPECT_EQ(p.pos_std, q.pos_std) << kClasses[c];
    EXPECT_EQ(p.vel_std, q.vel_std) << kClasses[c];
    EXPECT_EQ(p.stay_prob, q.stay_prob) << kClasses[c];
    EXPECT_EQ(p.init_omega_std, q.init_omega_std) << kClasses[c];
    ASSERT_EQ(p.models.size(), q.models.size()) << kClasses[c];
    for (size_t m = 0; m < p.models.size(); ++m) {
      EXPECT_EQ(p.models[m].kind, q.models[m].kind) << kClasses[c];
      EXPECT_EQ(p.models[m].q_acc, q.models[m].q_acc) << kClasses[c];
      EXPECT_EQ(p.models[m].q_omega, q.models[m].q_omega) << kClasses[c];
    }
  }
  EXPECT_EQ(t.classes, u.classes);
  expect_same(t.high_score, u.high_score, "high_score");
  EXPECT_EQ(t.low_score, u.low_score);
  EXPECT_EQ(t.matching, u.matching);
  EXPECT_EQ(t.use_velocity, u.use_velocity);
  EXPECT_EQ(t.tentative_max_age, u.tentative_max_age);
  EXPECT_EQ(t.output_coasting, u.output_coasting);
  expect_same(t.output_min_score, u.output_min_score, "output_min_score");
  EXPECT_EQ(t.history, u.history);
  EXPECT_EQ(t.size_alpha, u.size_alpha);
  EXPECT_EQ(t.z_alpha, u.z_alpha);
  EXPECT_EQ(t.yaw_alpha, u.yaw_alpha);
  EXPECT_EQ(t.heading_from_velocity, u.heading_from_velocity);
  expect_same(t.motion_on_speed, u.motion_on_speed, "motion_on_speed");
  expect_same(t.motion_off_speed, u.motion_off_speed, "motion_off_speed");
  EXPECT_EQ(t.motion_window, u.motion_window);
  EXPECT_EQ(t.motion_traj_window, u.motion_traj_window);
  EXPECT_EQ(t.motion_on_frames, u.motion_on_frames);
  EXPECT_EQ(t.motion_off_time, u.motion_off_time);
  EXPECT_EQ(t.stop_memory, u.stop_memory);
  EXPECT_EQ(cc.predictor.enabled, py.predictor.enabled);
  EXPECT_EQ(cc.predictor.horizon, py.predictor.horizon);
  EXPECT_EQ(cc.predictor.step, py.predictor.step);
  EXPECT_EQ(cc.predictor.merge_dist, py.predictor.merge_dist);
}

TEST(Config, MissingKeysAreErrors) {
  EXPECT_THROW(parse_config("version: 1\npreprocess: {}\n"), Error);
  EXPECT_THROW(parse_config("version: 2\n"), Error);
}

TEST(Localization, InspvaxMatchesPython) {
  REQUIRE_GOLDEN();
  const Npz z = load("ins.npz");
  const auto lat = member(z, "lat").to<double>(), lon = member(z, "lon").to<double>(), h = member(z, "h").to<double>();
  const auto az = member(z, "azimuth").to<double>();
  const EnuOrigin o(29.56, 106.45, 260.0);
  for (size_t k = 0; k < lat.size(); ++k) {
    EXPECT_LT((o.enu(lat[k], lon[k], h[k]) - vec3(member(z, "enu"), k)).cwiseAbs().maxCoeff(), 1e-6) << k;
    InsSample s;
    s.ins_status = kInsSolutionGood;
    s.latitude = lat[k];
    s.longitude = lon[k];
    s.height = h[k];
    s.azimuth = az[k];
    const auto p = pose_from_inspvax(s, o, Vec3(-0.06, -0.54, 0.0), 180.0);
    ASSERT_TRUE(p.has_value());
    EXPECT_LT((p->R - mat3(member(z, "R"), k)).cwiseAbs().maxCoeff(), 1e-14) << k;
    EXPECT_LT((p->p - vec3(member(z, "p"), k)).cwiseAbs().maxCoeff(), 1e-6) << k;
  }
  InsSample bad;
  bad.ins_status = 1;
  EXPECT_FALSE(pose_from_inspvax(bad, o, Vec3::Zero(), 0.0).has_value());
}

TEST(Association, LinearSumAssignmentMatchesScipy) {
  REQUIRE_GOLDEN();
  const Npz z = load("lsa.npz");
  for (int k = 0; k < 5; ++k) {
    const NpyArray& c = member(z, "cost" + std::to_string(k));
    Eigen::MatrixXd cost(c.dim(0), c.dim(1));
    for (int64_t i = 0; i < c.dim(0); ++i)
      for (int64_t j = 0; j < c.dim(1); ++j) cost(i, j) = c.at<double>(i * c.dim(1) + j);
    const auto rows = member(z, "rows" + std::to_string(k)).to<int>(), cols = member(z, "cols" + std::to_string(k)).to<int>();
    const Matches m = linear_sum_assignment(cost);
    ASSERT_EQ(m.size(), rows.size()) << k;
    for (size_t i = 0; i < m.size(); ++i) {
      EXPECT_EQ(m[i].first, rows[i]) << k;
      EXPECT_EQ(m[i].second, cols[i]) << k;
    }
  }
}

TEST(Association, GreedyTakesTheCheapestPairsFirst) {
  Eigen::MatrixXd c(3, 3);
  c << 1.0, 0.5, kInf, 0.2, 0.3, kInf, kInf, kInf, kInf;
  const Matches m = greedy_match(c, 1e9);
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(m[0], std::make_pair(1, 0));
  EXPECT_EQ(m[1], std::make_pair(0, 1));
}

}  // namespace dogperc::test
