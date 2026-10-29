// The online node's input handling against Python: PointCloud2 layouts (ros_utils.scan_from_msg), the
// node configuration, INSPVAX handling, and the sample bag's messages through OnlineInputs against
// dog_perception/online.py on the same messages (cpp_parity.py python --bag).
#include <filesystem>
#include <sstream>

#include "dogperc/message_log.hpp"
#include "dogperc/online.hpp"
#include "golden.hpp"

namespace dogperc::test {

namespace {

std::string text(const Npz& z, const std::string& key) {
  const NpyArray& a = member(z, key);
  return std::string(a.bytes.begin(), a.bytes.end());
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream in(s);
  for (std::string part; std::getline(in, part, sep);) out.push_back(part);
  return out;
}

}  // namespace

TEST(Online, CloudLayoutsMatchPython) {
  REQUIRE_GOLDEN();
  const Npz z = load("clouds.npz");
  const int n = static_cast<int>(member(z, "cases").scalar());
  ASSERT_GE(n, 4);
  for (int k = 0; k < n; ++k) {
    const std::string p = "c" + std::to_string(k) + "_";
    CloudView c;
    c.width = static_cast<uint32_t>(member(z, p + "width").scalar());
    c.height = static_cast<uint32_t>(member(z, p + "height").scalar());
    c.point_step = static_cast<uint32_t>(member(z, p + "point_step").scalar());
    c.is_bigendian = member(z, p + "bigendian").scalar() != 0;
    c.stamp = member(z, p + "stamp").scalar();
    const auto names = split(text(z, p + "field_names"), ',');
    const auto offs = member(z, p + "field_offsets").to<int64_t>();
    const auto types = member(z, p + "field_types").to<int64_t>();
    for (size_t f = 0; f < names.size(); ++f)
      c.fields.push_back({names[f], static_cast<uint32_t>(offs[f]), static_cast<uint8_t>(types[f]), 1});
    const NpyArray& data = member(z, p + "data");
    c.data = data.bytes.data();
    c.size = data.bytes.size();
    const LidarScan s = scan_from_cloud(c, text(z, p + "time_field"), text(z, p + "time_mode"),
                                        static_cast<int>(member(z, p + "mask").scalar()));
    EXPECT_EQ(s.xyz, member(z, p + "xyz").to<float>()) << "case " << k;
    EXPECT_EQ(s.intensity, member(z, p + "intensity").to<float>()) << "case " << k;
    EXPECT_EQ(s.point_times, member(z, p + "times").to<double>()) << "case " << k;
    EXPECT_EQ(s.stamp, member(z, p + "scan_stamp").scalar()) << "case " << k;
    EXPECT_GT(s.size(), 0u);
  }
}

TEST(Online, CloudErrors) {
  std::vector<uint8_t> bytes(12 * 4, 0);
  CloudView c;
  c.width = 4;
  c.height = 1;
  c.point_step = 12;
  c.fields = {{"x", 0, kFloat32, 1}, {"y", 4, kFloat32, 1}, {"z", 8, kFloat32, 1}};
  c.data = bytes.data();
  c.size = bytes.size() - 1;
  EXPECT_THROW(scan_from_cloud(c, "timestamp", "absolute"), Error);           // short data
  c.size = bytes.size();
  c.fields[2].offset = 10;
  EXPECT_THROW(scan_from_cloud(c, "timestamp", "absolute"), Error);           // z past point_step
  c.fields.pop_back();
  EXPECT_THROW(scan_from_cloud(c, "timestamp", "absolute"), Error);           // no z
}

TEST(Online, NodeConfigOfTheKygiPreset) {
  REQUIRE_GOLDEN();
  const NodeConfig c = load_node_config(golden("kygi.yaml"));
  EXPECT_EQ(c.lidar_topic, "/robot/topic/rdap/top_point_cloud2");
  EXPECT_EQ(c.localization, "inspvax");
  EXPECT_EQ(c.imu_convention, "kygi");
  EXPECT_EQ(c.time_mode, "absolute");
  EXPECT_EQ(c.ins_lever_arm, Vec3(-0.06, -0.54, 0.0));      // antennas swapped: the 0921 fit
  EXPECT_EQ(c.ins_azimuth_offset_deg, 180.0);
  EXPECT_THROW(load_node_config(golden("config.yaml")), Error);   // a sequence's config has no node: section
  EXPECT_NO_THROW(load_config(golden("kygi.yaml")));
}

TEST(Online, InspvaxWaitsForAGoodFix) {
  PipelineConfig cfg;
  PerceptionPipeline pipe(cfg);
  NodeConfig nc;
  nc.localization = "inspvax";
  std::vector<std::string> logged;
  OnlineInputs in(nc, pipe, [&](LogLevel, const std::string& t) { logged.push_back(t); });
  InsSample s;
  s.latitude = 29.7;
  s.longitude = 106.8;
  s.height = 217.0;
  s.ins_status = 1;                       // aligning
  s.stamp = 10.0;
  in.on_inspvax(s);
  EXPECT_FALSE(in.have_origin());
  EXPECT_EQ(pipe.poses().size(), 0u);
  s.ins_status = kInsSolutionGood;
  s.pos_type = kInsRtkFixed;
  for (int k = 0; k < 3; ++k) {
    s.stamp = 10.1 + 0.1 * k;
    in.on_inspvax(s);
  }
  EXPECT_TRUE(in.have_origin());
  EXPECT_EQ(pipe.poses().size(), 3u);
  s.ins_status = 2;                       // high variance: dropped
  s.stamp = 10.5;
  in.on_inspvax(s);
  EXPECT_EQ(pipe.poses().size(), 3u);
  ASSERT_GE(logged.size(), 3u);
  EXPECT_EQ(logged.front(), "waiting for a good INS solution");
  nc.localization = "gps";
  EXPECT_THROW(OnlineInputs(nc, pipe), Error);
}

TEST(Online, MessageLogMatchesPythonBagReplay) {
  REQUIRE_GOLDEN();
  if (!std::filesystem::exists(golden("online_msgs.npz")))
    GTEST_SKIP() << "no online_msgs.npz (dump_golden.py writes it with rosbags and samples/ present)";
  const PipelineConfig cfg = load_config(golden("online_config.yaml"));
  PerceptionPipeline pipe(cfg);
  OnlineInputs in(load_node_config(golden("online_config.yaml")), pipe);
  const MessageLog log(golden("online_msgs.npz"));
  const Npz py = load("online.npz");
  const Detections none;
  std::vector<PipelineOutput> frames;
  for (size_t i = 0; i < log.size(); ++i)
    if (auto o = log.feed(in, i, &none)) frames.push_back(std::move(*o));
  const auto stamps = member(py, "stamp").to<double>();
  ASSERT_EQ(frames.size(), stamps.size());
  const auto poff = member(py, "point_offsets").to<int64_t>();
  const NpyArray& pts = member(py, "points");
  double worst_pose = 0.0, worst_pt = 0.0;
  for (size_t f = 0; f < frames.size(); ++f) {
    const PipelineOutput& o = frames[f];
    EXPECT_EQ(o.stamp, stamps[f]);
    EXPECT_EQ(static_cast<int64_t>(o.frame.num_raw), member(py, "num_raw").at<int64_t>(f)) << f;
    EXPECT_EQ(static_cast<int64_t>(o.frame.num_kept), member(py, "num_kept").at<int64_t>(f)) << f;
    worst_pose = std::max(worst_pose, (o.frame.T_world_det - mat4(member(py, "T_world_det"), f)).cwiseAbs().maxCoeff());
    ASSERT_EQ(poff[f + 1] - poff[f], static_cast<int64_t>(o.frame.num_points())) << f;
    for (int64_t i = 0; i < 5 * (poff[f + 1] - poff[f]); ++i)
      worst_pt = std::max(worst_pt, static_cast<double>(std::abs(o.frame.points[i] - pts.at<float>(5 * poff[f] + i))));
  }
  EXPECT_LT(worst_pose, 1e-9);
  EXPECT_LT(worst_pt, 1e-4);
  std::printf("online parity: %zu frames from %zu messages, max diff poses %.2e, points %.2e m\n", frames.size(),
              log.size(), worst_pose, worst_pt);
}

}  // namespace dogperc::test
