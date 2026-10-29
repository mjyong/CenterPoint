#include "dogperc/message_log.hpp"

namespace dogperc {

namespace {

template <class T>
std::vector<T> col(const Npz& z, const std::string& name) {
  return member(z, name).to<T>();
}

}  // namespace

MessageLog::MessageLog(const std::string& path) : z_(read_npz(path)) {
  kind_ = col<int64_t>(z_, "order_kind");
  index_ = col<int64_t>(z_, "order_index");
  imu_t_ = col<double>(z_, "imu_t");
  imu_w_ = col<double>(z_, "imu_w");
  odom_t_ = col<double>(z_, "odom_t");
  odom_p_ = col<double>(z_, "odom_p");
  odom_q_ = col<double>(z_, "odom_q");
  odom_v_ = col<double>(z_, "odom_v");
  ins_t_ = col<double>(z_, "ins_t");
  ins_status_ = col<int64_t>(z_, "ins_status");
  ins_pos_type_ = col<int64_t>(z_, "ins_pos_type");
  ins_lla_ = col<double>(z_, "ins_lla");
  ins_azimuth_ = col<double>(z_, "ins_azimuth");
  ins_vel_ = col<double>(z_, "ins_vel");
  cloud_t_ = col<double>(z_, "cloud_t");
  cloud_width_ = col<int64_t>(z_, "cloud_width");
  cloud_height_ = col<int64_t>(z_, "cloud_height");
  cloud_step_ = col<int64_t>(z_, "cloud_point_step");
  cloud_big_ = col<int64_t>(z_, "cloud_bigendian");
  cloud_layout_ = col<int64_t>(z_, "cloud_layout");
  cloud_off_ = col<int64_t>(z_, "cloud_data_offsets");
  if (member(z_, "cloud_data").descr != "|u1") throw Error("message log: cloud_data must be uint8");
  const auto lo = col<int64_t>(z_, "layout_offsets"), fo = col<int64_t>(z_, "field_offset");
  const auto ft = col<int64_t>(z_, "field_datatype"), fc = col<int64_t>(z_, "field_count");
  const auto no = col<int64_t>(z_, "field_name_offsets");
  const NpyArray& names = member(z_, "field_names");
  for (size_t l = 0; l + 1 < lo.size(); ++l) {
    std::vector<CloudField> fields;
    for (int64_t f = lo[l]; f < lo[l + 1]; ++f) {
      CloudField cf;
      cf.name.assign(reinterpret_cast<const char*>(names.bytes.data()) + no[f], static_cast<size_t>(no[f + 1] - no[f]));
      cf.offset = static_cast<uint32_t>(fo[f]);
      cf.datatype = static_cast<uint8_t>(ft[f]);
      cf.count = static_cast<uint32_t>(fc[f]);
      fields.push_back(std::move(cf));
    }
    layouts_.push_back(std::move(fields));
  }
}

OdomMsg MessageLog::odom(size_t k) const {
  OdomMsg m;
  m.t = odom_t_[k];
  m.p = Vec3(odom_p_[3 * k], odom_p_[3 * k + 1], odom_p_[3 * k + 2]);
  m.q_xyzw = Eigen::Vector4d(odom_q_[4 * k], odom_q_[4 * k + 1], odom_q_[4 * k + 2], odom_q_[4 * k + 3]);
  m.v_linear = Vec3(odom_v_[3 * k], odom_v_[3 * k + 1], odom_v_[3 * k + 2]);
  return m;
}

InsSample MessageLog::inspvax(size_t k) const {
  InsSample s;
  s.stamp = ins_t_[k];
  s.ins_status = static_cast<int>(ins_status_[k]);
  s.pos_type = static_cast<int>(ins_pos_type_[k]);
  s.latitude = ins_lla_[3 * k];
  s.longitude = ins_lla_[3 * k + 1];
  s.height = ins_lla_[3 * k + 2];
  s.azimuth = ins_azimuth_[k];
  s.east_velocity = ins_vel_[3 * k];
  s.north_velocity = ins_vel_[3 * k + 1];
  s.up_velocity = ins_vel_[3 * k + 2];
  return s;
}

CloudView MessageLog::cloud(size_t k) const {
  CloudView c;
  c.stamp = cloud_t_[k];
  c.width = static_cast<uint32_t>(cloud_width_[k]);
  c.height = static_cast<uint32_t>(cloud_height_[k]);
  c.point_step = static_cast<uint32_t>(cloud_step_[k]);
  c.is_bigendian = cloud_big_[k] != 0;
  c.fields = layouts_.at(static_cast<size_t>(cloud_layout_[k]));
  c.data = member(z_, "cloud_data").bytes.data() + cloud_off_[k];
  c.size = static_cast<size_t>(cloud_off_[k + 1] - cloud_off_[k]);
  return c;
}

std::optional<PipelineOutput> MessageLog::feed(OnlineInputs& in, size_t i, const Detections* detections) const {
  const size_t k = index(i);
  switch (kind(i)) {
    case kImu:
      in.on_imu(imu_t(k), imu_w(k));
      return std::nullopt;
    case kOdom:
      in.on_odom(odom(k));
      return std::nullopt;
    case kInspvax:
      in.on_inspvax(inspvax(k));
      return std::nullopt;
    default:
      return in.on_cloud(cloud(k), detections);
  }
}

}  // namespace dogperc
