// The detector's host steps against Python: model input, voxels (identical), PFN input, BEV canvas, decoding,
// class mapping and NMS, the geometric filters, and the whole PillarDetector around a replayed network.
#include "lidar3d/detection/detector.hpp"
#include "lidar3d/detection/filters.hpp"
#include "golden.hpp"

namespace lidar3d::test {

namespace {

std::vector<std::vector<float>> golden_heads(const Npz& d, const ModelMeta& m) {
  std::vector<std::vector<float>> heads;
  for (int t = 0; t < m.num_tasks(); ++t)
    for (const auto& k : m.head_keys) heads.push_back(member(d, "head" + std::to_string(t) + "_" + k).to<float>());
  return heads;
}

// returns the Python stages instead of running a network
class ReplayNetwork : public PillarNetwork {
 public:
  ReplayNetwork(std::vector<float> pfn_out, std::vector<std::vector<float>> heads)
      : pfn_out_(std::move(pfn_out)), heads_(std::move(heads)) {}
  std::vector<float> pfn(const std::vector<float>&, int) override { return pfn_out_; }
  std::vector<std::vector<float>> rpn(const std::vector<float>&) override { return heads_; }

 private:
  std::vector<float> pfn_out_;
  std::vector<std::vector<float>> heads_;
};

}  // namespace

TEST(Detection, ModelMetaLoads) {
  REQUIRE_GOLDEN();
  const ModelMeta m = load_model_meta(golden("model/model_meta.json"));
  EXPECT_EQ(m.num_tasks(), 6);
  EXPECT_EQ(m.grid_size[0], 204);
  EXPECT_EQ(m.rpn_outputs.front(), "task0_reg");
  EXPECT_EQ(m.head_channels(1, "hm"), 2);
  EXPECT_TRUE(m.merged_nms[1].set);
  EXPECT_EQ(m.merged_nms[1].weak_radius, 1.2);
  EXPECT_EQ(grid_size(m.voxel_size, m.pc_range), m.grid_size);
}

TEST(Detection, VoxelsAreIdentical) {
  REQUIRE_GOLDEN();
  const ModelMeta m = load_model_meta(golden("model/model_meta.json"));
  const Npz z = load("pillars.npz");
  PillarDetector det(m, std::make_unique<ReplayNetwork>(std::vector<float>{}, std::vector<std::vector<float>>{}), {});
  const auto pts = det.model_input(member(z, "points_det").to<float>());
  EXPECT_EQ(pts, member(z, "model_points").to<float>());
  const Voxels v = voxelize(pts.data(), pts.size() / 5, 5, m.voxel_size, m.pc_range, m.max_points, m.max_voxels);
  ASSERT_EQ(v.num_voxels, member(z, "num_points").dim(0));
  EXPECT_EQ(v.voxels, member(z, "voxels").to<float>());
  EXPECT_EQ(v.coords, member(z, "coords").to<int32_t>());
  EXPECT_EQ(v.num_points, member(z, "num_points").to<int32_t>());

  const auto feats = decorate_pillars(v, m.voxel_size, m.pc_range);
  const auto want = member(z, "feats").to<float>();
  ASSERT_EQ(feats.size(), want.size());
  double worst = 0.0;
  for (size_t i = 0; i < feats.size(); ++i) worst = std::max(worst, static_cast<double>(std::abs(feats[i] - want[i])));
  EXPECT_LT(worst, 1e-6);
  // the rest of the chain from the Python inputs, so a decorate rounding difference does not carry over
  const auto pfn_in = pad_pillars(want, v.num_voxels, m.max_points, 10, m.max_pillars);
  EXPECT_EQ(pfn_in, member(z, "pfn_in").to<float>());
  const auto pf = member(z, "pfn_out").to<float>();
  const auto bev = scatter_pillars(pf.data(), m.bev_channels, m.max_pillars, v.coords,
                                   static_cast<int>(member(z, "num_valid").scalar()), m.grid_size[0], m.grid_size[1]);
  EXPECT_EQ(bev, member(z, "bev").to<float>());
}

TEST(Detection, DecodePostprocessFiltersMatchPython) {
  REQUIRE_GOLDEN();
  const ModelMeta m = load_model_meta(golden("model/model_meta.json"));
  const Npz d = load("decode.npz");
  const RawBoxes raw = decode_centerpoint(split_head_outputs(golden_heads(d, m), m), m);
  const NpyArray& box9 = member(d, "box9");
  ASSERT_EQ(static_cast<int64_t>(raw.scores.size()), box9.dim(0));
  const auto sc = member(d, "scores").to<double>();
  const auto lb = member(d, "labels").to<int>();
  for (size_t i = 0; i < raw.scores.size(); ++i) {
    EXPECT_EQ(raw.labels[i], lb[i]) << i;
    EXPECT_NEAR(raw.scores[i], sc[i], 1e-6) << i;
    for (int k = 0; k < 9; ++k) EXPECT_NEAR(raw.box9[i][k], box9.at<double>(9 * i + k), 1e-5) << i << " " << k;
  }
  const Detections dets = postprocess(raw, m, 1.5);
  expect_same_detections(dets, detections(d, "raw", 0), 1e-5, "postprocess");

  const auto pts = member(d, "points_det").to<float>();
  const auto why = filter_reasons(dets, pts.data(), pts.size() / 5, 5, DetectionFilterConfig{});
  const auto want = member(d, "reasons").to<int>();
  const char* names[] = {"", "floating", "sunk", "few points", "duplicate"};
  ASSERT_EQ(why.size(), want.size());
  for (size_t i = 0; i < why.size(); ++i) EXPECT_EQ(why[i], names[want[i]]) << "box " << i;
  std::vector<double> bottom;
  std::vector<int64_t> counts;
  filter_measurements(dets, pts.data(), pts.size() / 5, 5, DetectionFilterConfig{}, bottom, counts);
  // the boxes come from the C++ decoder: float32 exp / atan2 may differ from NumPy's by an ulp
  EXPECT_LT(max_abs(bottom, member(d, "bottom").to<double>()), 1e-6);
  EXPECT_EQ(counts, member(d, "counts").to<int64_t>());
  const Detections final = filter_detections(dets, pts.data(), pts.size() / 5, 5, DetectionFilterConfig{});
  expect_same_detections(final, detections(d, "final", 0), 1e-5, "filters");
}

TEST(Detection, PillarDetectorAroundAReplayedNetwork) {
  REQUIRE_GOLDEN();
  const ModelMeta m = load_model_meta(golden("model/model_meta.json"));
  const Npz d = load("decode.npz");
  const Npz p = load("pillars.npz");
  PillarDetector det(m, std::make_unique<ReplayNetwork>(member(p, "pfn_out").to<float>(), golden_heads(d, m)),
                     DetectionFilterConfig{});
  det.keep_stages = true;
  const Detections out = det.detect(member(d, "points_det").to<float>(), 1.5);
  expect_same_detections(out, detections(d, "final", 0), 1e-5, "PillarDetector");
  expect_same_detections(det.last_raw(), detections(d, "raw", 0), 1e-5, "PillarDetector raw");
  EXPECT_EQ(det.stages().bev, member(p, "bev").to<float>());
  EXPECT_EQ(det.last_pillars().first, det.last_pillars().second);
  EXPECT_TRUE(det.last_timing().count("network_ms"));
}

}  // namespace lidar3d::test
