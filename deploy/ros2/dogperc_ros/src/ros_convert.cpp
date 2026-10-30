#include "ros_convert.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace dogperc::ros {

double seconds(const builtin_interfaces::msg::Time& t) {
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

builtin_interfaces::msg::Time stamp(double t) {
  builtin_interfaces::msg::Time s;
  double sec = std::floor(t);
  int64_t ns = std::llround((t - sec) * 1e9);
  if (ns >= 1000000000) {
    sec += 1.0;
    ns -= 1000000000;
  }
  s.sec = static_cast<int32_t>(sec);
  s.nanosec = static_cast<uint32_t>(ns);
  return s;
}

CloudView cloud_view(const sensor_msgs::msg::PointCloud2& m) {
  CloudView c;
  c.stamp = seconds(m.header.stamp);
  c.width = m.width;
  c.height = m.height;
  c.point_step = m.point_step;
  c.is_bigendian = m.is_bigendian;
  for (const auto& f : m.fields) c.fields.push_back({f.name, f.offset, f.datatype, f.count});
  c.data = m.data.data();
  c.size = m.data.size();
  return c;
}

Vec3 angular_velocity(const sensor_msgs::msg::Imu& m) {
  return Vec3(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z);
}

OdomMsg odom(const nav_msgs::msg::Odometry& m) {
  OdomMsg o;
  o.t = seconds(m.header.stamp);
  const auto& p = m.pose.pose.position;
  const auto& q = m.pose.pose.orientation;
  const auto& v = m.twist.twist.linear;
  o.p = Vec3(p.x, p.y, p.z);
  o.q_xyzw = Eigen::Vector4d(q.x, q.y, q.z, q.w);
  o.v_linear = Vec3(v.x, v.y, v.z);
  return o;
}

#ifdef DOGPERC_HAVE_KYGI
InsSample ins_sample(const kygi665_msgs::msg::INSPVAX& m) {
  InsSample s;
  s.stamp = seconds(m.header.stamp);
  s.ins_status = static_cast<int>(m.ins_status.data);
  s.pos_type = static_cast<int>(m.pos_type.data);
  s.latitude = m.latitude;
  s.longitude = m.longitude;
  s.height = m.height;
  s.azimuth = m.azimuth;
  s.east_velocity = m.east_velocity;
  s.north_velocity = m.north_velocity;
  s.up_velocity = m.up_velocity;
  return s;
}
#endif

namespace {

geometry_msgs::msg::Point point(double x, double y, double z) {
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

geometry_msgs::msg::Vector3 vector3(double x, double y, double z) {
  geometry_msgs::msg::Vector3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

struct Rgb {
  float r, g, b;
};
Rgb color(int label) {
  static const Rgb colors[] = {{0.2f, 0.6f, 1.0f}, {1.0f, 0.3f, 0.3f}, {1.0f, 0.8f, 0.1f}};   // vehicle, pedestrian, cyclist
  return label >= 0 && label < 3 ? colors[label] : Rgb{1.0f, 1.0f, 1.0f};
}

visualization_msgs::msg::Marker delete_all() {
  visualization_msgs::msg::Marker m;
  m.action = visualization_msgs::msg::Marker::DELETEALL;
  return m;
}

}  // namespace

dog_perception_msgs::msg::ObjectArray object_array(const PerceptionPipeline& pipe, const PipelineOutput& out,
                                                   const std::string& local_frame, const std::string& world_frame) {
  dog_perception_msgs::msg::ObjectArray arr;
  arr.header.stamp = stamp(out.stamp);
  arr.header.frame_id = local_frame;
  arr.world_frame = world_frame;
  const Mat4& T = out.frame.T_world_det;
  arr.world_pose.position = point(T(0, 3), T(1, 3), T(2, 3));
  const Eigen::Vector4d q = R_to_quat(T.topLeftCorner<3, 3>());   // x, y, z, w
  arr.world_pose.orientation.x = q[0];
  arr.world_pose.orientation.y = q[1];
  arr.world_pose.orientation.z = q[2];
  arr.world_pose.orientation.w = q[3];
  for (const PlannerObject& po : planner_objects(pipe, out)) {
    const LocalTrack& t = po.track;
    dog_perception_msgs::msg::Object o;
    o.id = static_cast<uint32_t>(t.id);
    o.label = static_cast<uint8_t>(t.label);
    o.label_name = kClasses[t.label];
    o.motion = static_cast<uint8_t>(t.motion);
    o.score = static_cast<float>(t.score);
    o.coasting = t.coasting;
    o.age = static_cast<float>(t.age);
    o.position = point(t.position[0], t.position[1], t.position[2]);
    o.yaw = t.yaw;
    o.size = vector3(t.size[0], t.size[1], t.size[2]);
    o.velocity = vector3(t.velocity[0], t.velocity[1], 0.0);
    for (int k = 0; k < 16; ++k) o.covariance[k] = t.covariance(k / 4, k % 4);
    for (double s : po.times) o.prediction_times.push_back(static_cast<float>(s));
    for (const auto& path : po.paths) {
      dog_perception_msgs::msg::PredictedPath p;
      p.probability = static_cast<float>(path.probability);
      for (size_t s = 0; s < path.xy.size(); ++s) {
        p.positions.push_back(point(path.xy[s][0], path.xy[s][1], 0.0));
        for (int k = 0; k < 4; ++k) p.covariance.push_back(path.covariance[s](k / 2, k % 2));
      }
      o.predictions.push_back(std::move(p));
    }
    arr.objects.push_back(std::move(o));
  }
  return arr;
}

visualization_msgs::msg::MarkerArray track_markers(const PipelineOutput& out, const std::string& world_frame) {
  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(delete_all());
  for (const TrackState& s : out.tracks) {
    const Rgb c = color(s.label);
    visualization_msgs::msg::Marker m;
    m.header.frame_id = world_frame;
    m.header.stamp = stamp(out.stamp);
    m.ns = "box";
    m.id = s.track_id;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.pose.position = point(s.position[0], s.position[1], s.position[2]);
    m.pose.orientation.z = std::sin(s.yaw / 2);
    m.pose.orientation.w = std::cos(s.yaw / 2);
    m.scale = vector3(s.size[0], s.size[1], s.size[2]);
    m.color.r = c.r;
    m.color.g = c.g;
    m.color.b = c.b;
    m.color.a = s.coasting ? 0.25f : 0.6f;
    arr.markers.push_back(m);
    visualization_msgs::msg::Marker txt;
    txt.header = m.header;
    txt.ns = "id";
    txt.id = s.track_id;
    txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    txt.pose.position = point(s.position[0], s.position[1], s.position[2] + s.size[2]);
    txt.pose.orientation.w = 1.0;
    txt.scale.z = 0.5;
    txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0f;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%.3s %d %.1fm/s %s", s.name(), s.track_id, std::hypot(s.velocity[0], s.velocity[1]),
                  motion_name(s.motion));
    txt.text = buf;
    arr.markers.push_back(txt);
  }
  return arr;
}

visualization_msgs::msg::MarkerArray prediction_markers(const PipelineOutput& out, const std::string& world_frame) {
  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(delete_all());
  int k = 0;
  for (const Prediction& p : out.predictions)
    for (size_t j = 0; j < p.modes.size(); ++j) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = world_frame;
      m.header.stamp = stamp(out.stamp);
      m.ns = "pred";
      m.id = k++;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.pose.orientation.w = 1.0;
      m.scale.x = 0.08;
      m.color.r = 1.0f;
      m.color.g = 0.2f;
      m.color.b = 0.2f;
      m.color.a = static_cast<float>(0.2 + 0.8 * p.probs[j]);
      for (const Vec2& xy : p.modes[j]) m.points.push_back(point(xy[0], xy[1], 0.2));
      arr.markers.push_back(m);
    }
  return arr;
}

sensor_msgs::msg::PointCloud2 det_cloud(const PipelineOutput& out, const std::string& local_frame) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.frame_id = local_frame;
  msg.header.stamp = stamp(out.stamp);
  const size_t n = out.frame.num_points();
  msg.height = 1;
  msg.width = static_cast<uint32_t>(n);
  const char* names[] = {"x", "y", "z", "intensity", "dt"};
  for (uint32_t i = 0; i < 5; ++i) {
    sensor_msgs::msg::PointField f;
    f.name = names[i];
    f.offset = 4 * i;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    msg.fields.push_back(f);
  }
  msg.point_step = 20;
  msg.row_step = static_cast<uint32_t>(20 * n);
  msg.is_bigendian = false;
  msg.is_dense = true;
  msg.data.resize(20 * n);
  std::memcpy(msg.data.data(), out.frame.points.data(), msg.data.size());
  return msg;
}

namespace {

geometry_msgs::msg::TransformStamped transform(const Mat4& T, double t, const std::string& parent,
                                               const std::string& child) {
  geometry_msgs::msg::TransformStamped m;
  m.header.stamp = stamp(t);
  m.header.frame_id = parent;
  m.child_frame_id = child;
  m.transform.translation.x = T(0, 3);
  m.transform.translation.y = T(1, 3);
  m.transform.translation.z = T(2, 3);
  const Eigen::Vector4d q = R_to_quat(T.topLeftCorner<3, 3>());   // x, y, z, w
  m.transform.rotation.x = q[0];
  m.transform.rotation.y = q[1];
  m.transform.rotation.z = q[2];
  m.transform.rotation.w = q[3];
  return m;
}

}  // namespace

std::vector<geometry_msgs::msg::TransformStamped> viewer_transforms(const PerceptionPipeline& pipe,
                                                                    const PipelineOutput& out,
                                                                    const sensor_msgs::msg::PointCloud2& cloud,
                                                                    const std::string& local_frame,
                                                                    const std::string& world_frame) {
  std::vector<geometry_msgs::msg::TransformStamped> tf{transform(out.frame.T_world_det, out.stamp, world_frame,
                                                                 local_frame)};
  if (cloud.header.frame_id.empty() || cloud.header.frame_id == local_frame) return tf;
  const double t = seconds(cloud.header.stamp);
  try {
    tf.push_back(transform(pipe.poses().pose_at(t) * pipe.config().preprocess.T_body_lidar, t, world_frame,
                           cloud.header.frame_id));
  } catch (const PoseOutOfRange&) {
  }
  return tf;
}

}  // namespace dogperc::ros
