// The whole pipeline on the golden recording against PerceptionPipeline (same detections fed in):
// det-frame clouds, poses, ground estimate, tracks (identical IDs, states 1e-6) and predictions.
#include <cstdlib>
#include <filesystem>

#include "dogperc/parallel.hpp"
#include "dogperc/pipeline.hpp"
#include "dogperc/sequence.hpp"
#include "golden.hpp"

namespace dogperc::test {

namespace {

struct CppRun {
  std::vector<PipelineOutput> frames;
  std::vector<double> ground_offset;
  std::vector<std::array<double, 2>> ground_tilt;
};

CppRun run_cpp(const Npz& dets, double odom_latency) {
  const PipelineConfig cfg = load_config(golden("config.yaml"));
  PerceptionPipeline pipe(cfg);
  const SequenceReader reader(golden("seq"));
  CppRun run;
  size_t k = 0;
  const auto stamps = member(dets, "stamp").to<double>();
  for (const auto& ev : reader.events(odom_latency)) {
    if (const auto* i = std::get_if<ImuEvent>(&ev)) {
      pipe.on_imu(i->t, i->gyro);
    } else if (const auto* o = std::get_if<OdomEvent>(&ev)) {
      pipe.on_odometry(o->t, o->R, o->p, o->v);
    } else {
      const auto& s = std::get<ScanEvent>(ev);
      // Python fed detections to the scans it produced a frame for, in order
      if (k >= stamps.size() || std::abs(stamps[k] - s.stamp) > 1e-9) {
        const Detections none;
        const auto out = pipe.on_scan(reader.load_scan(s.index), &none);
        EXPECT_FALSE(out.has_value()) << "C++ made a frame at " << s.stamp << " where Python made none";
        continue;
      }
      const Detections d = detections(dets, "", static_cast<int64_t>(k));
      auto out = pipe.on_scan(reader.load_scan(s.index), &d);
      EXPECT_TRUE(out.has_value()) << "no frame at " << s.stamp;
      if (!out) continue;
      run.frames.push_back(std::move(*out));
      run.ground_offset.push_back(pipe.preprocessor().ground() ? pipe.preprocessor().ground()->offset : 0.0);
      run.ground_tilt.push_back(pipe.preprocessor().ground() ? pipe.preprocessor().ground()->tilt : std::array<double, 2>{});
      ++k;
    }
  }
  return run;
}

}  // namespace

TEST(Pipeline, MatchesPythonOnTheGoldenRecording) {
  REQUIRE_GOLDEN();
  const Npz py = load("pipeline.npz");
  const Npz dets = load("dets.npz");
  const CppRun run = run_cpp(dets, member(py, "odom_latency").scalar());
  const auto stamp = member(py, "stamp").to<double>();
  ASSERT_EQ(run.frames.size(), stamp.size());
  const auto poff = member(py, "point_offsets").to<int64_t>();
  const NpyArray& pts = member(py, "points");
  const auto toff = member(py, "trk_offsets").to<int64_t>();
  const auto qoff = member(py, "pred_offsets").to<int64_t>();
  double worst_pt = 0.0, worst_trk = 0.0, worst_pred = 0.0, worst_pose = 0.0;
  for (size_t f = 0; f < run.frames.size(); ++f) {
    const PipelineOutput& o = run.frames[f];
    EXPECT_EQ(o.stamp, stamp[f]);
    worst_pose = std::max(worst_pose, (o.frame.T_world_det - mat4(member(py, "T_world_det"), f)).cwiseAbs().maxCoeff());
    worst_pose = std::max(worst_pose, (o.frame.T_world_body - mat4(member(py, "T_world_body"), f)).cwiseAbs().maxCoeff());
    EXPECT_NEAR(run.ground_offset[f], member(py, "ground_offset").at<double>(f), 1e-9) << f;
    for (int k = 0; k < 2; ++k) EXPECT_NEAR(run.ground_tilt[f][k], member(py, "ground_tilt").at<double>(2 * f + k), 1e-9) << f;
    EXPECT_EQ(static_cast<int64_t>(o.frame.num_kept), member(py, "num_kept").at<int64_t>(f)) << f;
    // det-frame cloud: same points in the same order (the golden data keeps the clouds of some frames)
    ASSERT_EQ(static_cast<int64_t>(o.frame.num_points()), member(py, "num_points").at<int64_t>(f)) << "frame " << f;
    if (poff[f + 1] > poff[f]) {
      ASSERT_EQ(poff[f + 1] - poff[f], static_cast<int64_t>(o.frame.num_points()));
    }
    for (int64_t i = 0; i < poff[f + 1] - poff[f]; ++i)
      for (int k = 0; k < 5; ++k)
        worst_pt = std::max(worst_pt, static_cast<double>(std::abs(o.frame.points[5 * i + k] - pts.at<float>((poff[f] + i) * 5 + k))));
    // tracks: identical IDs and order
    ASSERT_EQ(static_cast<int64_t>(o.tracks.size()), toff[f + 1] - toff[f]) << "frame " << f;
    for (size_t j = 0; j < o.tracks.size(); ++j) {
      const TrackState& t = o.tracks[j];
      const int64_t r = toff[f] + j;
      ASSERT_EQ(t.track_id, member(py, "trk_id").at<int>(r)) << "frame " << f;
      EXPECT_EQ(t.label, member(py, "trk_label").at<int>(r));
      EXPECT_EQ(static_cast<int>(t.motion), member(py, "trk_motion").at<int>(r)) << "frame " << f << " track " << t.track_id;
      EXPECT_EQ(t.coasting, member(py, "trk_coasting").at<int>(r) != 0);
      EXPECT_EQ(t.hits, member(py, "trk_hits").at<int>(r));
      const double d[] = {
          (t.position - vec3(member(py, "trk_position"), r)).cwiseAbs().maxCoeff(),
          std::abs(t.velocity[0] - member(py, "trk_velocity").at<double>(2 * r)),
          std::abs(t.velocity[1] - member(py, "trk_velocity").at<double>(2 * r + 1)),
          std::abs(wrap_angle(t.yaw - member(py, "trk_yaw").at<double>(r))),
          (t.size - vec3(member(py, "trk_size"), r)).cwiseAbs().maxCoeff(),
          std::abs(t.score - member(py, "trk_score").at<double>(r)),
          std::abs(t.age - member(py, "trk_age").at<double>(r)),
          std::abs(t.mode_probs[0] - member(py, "trk_mode_probs").at<double>(2 * r)),
      };
      for (double v : d) worst_trk = std::max(worst_trk, v);
      for (int k = 0; k < 16; ++k)
        worst_trk = std::max(worst_trk, std::abs(t.covariance(k / 4, k % 4) - member(py, "trk_covariance").at<double>(16 * r + k)));
      EXPECT_EQ(t.history.size(), static_cast<size_t>(member(py, "trk_history_len").at<int64_t>(r)));
      EXPECT_EQ(t.has_measurement, !std::isnan(member(py, "trk_measurement").at<double>(6 * r)));
    }
    ASSERT_EQ(static_cast<int64_t>(o.predictions.size()), qoff[f + 1] - qoff[f]) << "frame " << f;
    const NpyArray& pm = member(py, "pred_modes");
    const int64_t T = pm.dim(2);
    for (size_t j = 0; j < o.predictions.size(); ++j) {
      const Prediction& p = o.predictions[j];
      const int64_t r = qoff[f] + j;
      EXPECT_EQ(p.track_id, member(py, "pred_track_id").at<int>(r));
      ASSERT_EQ(static_cast<int64_t>(p.probs.size()), member(py, "pred_num_modes").at<int64_t>(r)) << "frame " << f;
      for (size_t k = 0; k < p.probs.size(); ++k) {
        worst_pred = std::max(worst_pred, std::abs(p.probs[k] - member(py, "pred_probs").at<double>(2 * r + k)));
        for (int64_t s = 0; s < T; ++s)
          for (int c = 0; c < 2; ++c) {
            worst_pred = std::max(worst_pred, std::abs(p.modes[k][s][c] - pm.at<double>(((r * 2 + k) * T + s) * 2 + c)));
            for (int e = 0; e < 2; ++e)
              worst_pred = std::max(worst_pred, std::abs(p.covs[k][s](c, e) - member(py, "pred_covs").at<double>((((r * 2 + k) * T + s) * 2 + c) * 2 + e)));
          }
      }
    }
  }
  // docs/DEPLOY_ORIN.md section 6: points 1e-4 m, track states 1e-6, predictions 1e-4
  EXPECT_LT(worst_pose, 1e-9);
  EXPECT_LT(worst_pt, 1e-4);
  EXPECT_LT(worst_trk, 1e-6);
  EXPECT_LT(worst_pred, 1e-4);
  RecordProperty("max_point_diff", std::to_string(worst_pt));
  RecordProperty("max_track_diff", std::to_string(worst_trk));
  RecordProperty("max_prediction_diff", std::to_string(worst_pred));
  std::printf("pipeline parity: %zu frames, max diff points %.2e m, tracks %.2e, predictions %.2e, poses %.2e\n",
              run.frames.size(), worst_pt, worst_trk, worst_pred, worst_pose);
}

TEST(Pipeline, ThreadCountDoesNotChangeTheOutput) {
  REQUIRE_GOLDEN();
  const Npz py = load("pipeline.npz");
  const Npz dets = load("dets.npz");
  const double latency = member(py, "odom_latency").scalar();
  set_num_threads(1);
  const CppRun one = run_cpp(dets, latency);
  set_num_threads(4);
  const CppRun four = run_cpp(dets, latency);
  set_num_threads(1);
  ASSERT_EQ(one.frames.size(), four.frames.size());
  for (size_t f = 0; f < one.frames.size(); ++f) {
    EXPECT_EQ(one.frames[f].frame.points, four.frames[f].frame.points) << "frame " << f;
    ASSERT_EQ(one.frames[f].tracks.size(), four.frames[f].tracks.size());
    for (size_t j = 0; j < one.frames[f].tracks.size(); ++j)
      EXPECT_EQ(one.frames[f].tracks[j].position, four.frames[f].tracks[j].position);
  }
}

TEST(Pipeline, ReplayToolWritesOneRowPerFrame) {
  REQUIRE_GOLDEN();
  const std::string out = (std::filesystem::temp_directory_path() / "dogperc_replay_test.npz").string();
  const std::string cmd = std::string(DOGPERC_REPLAY_BIN) + " --seq " + golden("seq") + " --config " + golden("config.yaml") +
                          " --detections " + golden("dets.npz") + " --odom-latency 0.05 --out " + out + " > /dev/null";
  ASSERT_EQ(std::system(cmd.c_str()), 0) << cmd;
  const Npz z = read_npz(out);
  const Npz py = load("pipeline.npz");
  EXPECT_EQ(member(z, "stamp").to<double>(), member(py, "stamp").to<double>());
  EXPECT_EQ(member(z, "trk_id").to<int64_t>(), member(py, "trk_id").to<int64_t>());
  EXPECT_EQ(member(z, "trk_offsets").to<int64_t>(), member(py, "trk_offsets").to<int64_t>());
  std::remove(out.c_str());
}

}  // namespace dogperc::test
