#include "lidar3d/tracking/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "lidar3d/tracking/association.hpp"

namespace lidar3d {

const char* motion_name(Motion m) {
  switch (m) {
    case Motion::Moving: return "moving";
    case Motion::Stopped: return "stopped";
    default: return "static";
  }
}

namespace {

// NumPy's pairwise summation of a 1-D array (np.add.reduce): blocks of 8 partial sums below 128
double np_sum(const std::vector<double>& a, size_t lo, size_t n) {
  if (n < 8) {
    double r = 0.0;
    for (size_t i = 0; i < n; ++i) r += a[lo + i];
    return r;
  }
  if (n <= 128) {
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[lo + k];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int k = 0; k < 8; ++k) r[k] += a[lo + i + k];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[lo + i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return np_sum(a, lo, n2) + np_sum(a, lo + n2, n - n2);
}

bool center_in_box(double x, double y, double cx, double cy, double l, double w, double yaw) {
  const double dx = x - cx, dy = y - cy;
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
  return std::abs(lx) <= l / 2 && std::abs(ly) <= w / 2;
}

// same class, and one center inside the other box: one long vehicle, not a neighbour
bool second_box_on_track(const Box& box, int label, const Track& t) {
  if (t.label != label) return false;
  const StateVec x = t.imm.mean();
  if (center_in_box(box[0], box[1], x[0], x[1], t.size[0], t.size[1], t.yaw)) return true;
  return center_in_box(x[0], x[1], box[0], box[1], box[3], box[4], box[6]);
}

}  // namespace

Track::Track(int track_id, int lab, const Box& box, const std::array<double, 2>* vel, double sc, double st,
             std::shared_ptr<const TrackerConfig> shared_cfg)
    : id(track_id), label(lab), cfg_(std::move(shared_cfg)) {
  const TrackerConfig& cfg = *cfg_;
  const ClassParams& params = cfg.class_params[label];
  std::vector<MotionModel> models;
  for (const auto& s : params.models) models.push_back(MotionModel::from_spec(s));
  imm = IMM(models, params.stay_prob);
  std::array<double, 2> v{0.0, 0.0};
  if (cfg.use_velocity && vel) v = *vel;
  const double v_var = cfg.use_velocity ? std::pow(params.vel_std, 2.0) : 4.0;
  StateVec x0;
  x0 << box[0], box[1], v[0], v[1], 0.0;
  StateVec d;
  d << std::pow(params.pos_std, 2.0), std::pow(params.pos_std, 2.0), v_var, v_var, std::pow(params.init_omega_std, 2.0);
  imm.initialize(x0, d.asDiagonal());
  z = box[2];
  size = Vec3(box[3], box[4], box[5]);
  yaw = box[6];
  score = sc;
  birth = last_update = stamp = st;
  hits = 1;
  confirmed = params.min_hits <= 1;
  add_center(box);
  measure(box, vel, sc);
  record(true);
  update_motion(cfg.use_velocity ? vel : nullptr);
}

void Track::predict(double st) {
  const double dt = st - stamp;
  if (dt > 0) {
    imm.predict(dt);
    stamp = st;
  }
}

double Track::gate() const {
  StateVec x;
  StateMat P;
  imm.state(x, P);
  const double sigma = std::sqrt(std::max((P(0, 0) + P(1, 1)) / 2, 0.0));
  return std::clamp(3 * sigma, params().gate, 3 * params().gate);
}

void Track::update(const Box& box, const std::array<double, 2>* vel, double sc) {
  const ClassParams& p = params();
  const TrackerConfig& cfg = *cfg_;
  if (cfg.use_velocity && vel) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 5);
    H.leftCols(4).setIdentity();
    Eigen::VectorXd zz(4);
    zz << box[0], box[1], (*vel)[0], (*vel)[1];
    Eigen::VectorXd r(4);
    r << std::pow(p.pos_std, 2.0), std::pow(p.pos_std, 2.0), std::pow(p.vel_std, 2.0), std::pow(p.vel_std, 2.0);
    imm.update(zz, H, r.asDiagonal().toDenseMatrix());
  } else {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 5);
    H.leftCols(2).setIdentity();
    Eigen::VectorXd zz(2);
    zz << box[0], box[1];
    imm.update(zz, H, Eigen::MatrixXd::Identity(2, 2) * std::pow(p.pos_std, 2.0));
  }
  z += cfg.z_alpha * (box[2] - z);
  for (int k = 0; k < 3; ++k) size[k] += cfg.size_alpha * (box[3 + k] - size[k]);
  update_yaw(box[6]);
  score = 0.7 * score + 0.3 * sc;
  hits += 1;
  last_update = stamp;
  if (hits >= p.min_hits) confirmed = true;
  record(true);
  add_center(box);
  measure(box, vel, sc);
  update_motion(cfg.use_velocity ? vel : nullptr);
}

void Track::measure(const Box& box, const std::array<double, 2>* vel, double sc) {
  meas_ = {box[0], box[1], box[6], vel ? (*vel)[0] : kNaN, vel ? (*vel)[1] : kNaN, sc};
}

void Track::add_center(const Box& box) {
  centers_.push_back({stamp, box[0], box[1]});
  while (centers_.front()[0] < stamp - cfg_->motion_traj_window - 1e-6) centers_.pop_front();
}

std::optional<double> Track::displacement_speed() const {
  const size_t n = centers_.size();
  if (n < 3 || centers_.back()[0] - centers_.front()[0] < 0.3 - 1e-6) return std::nullopt;
  std::vector<double> t(n), x(n), y(n);
  for (size_t i = 0; i < n; ++i) {
    t[i] = centers_[i][0];
    x[i] = centers_[i][1];
    y[i] = centers_[i][2];
  }
  // c[:, 0].mean() sums a 1-D column (pairwise); c[:, 1:].mean(0) sums along axis 0 (row by row)
  const double tm = np_sum(t, 0, n) / static_cast<double>(n);
  double xs = 0.0, ys = 0.0;
  for (size_t i = 0; i < n; ++i) {
    xs += x[i];
    ys += y[i];
  }
  const double xm = xs / static_cast<double>(n), ym = ys / static_cast<double>(n);
  double tt = 0.0, tx = 0.0, ty = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double ti = t[i] - tm;
    tt += ti * ti;
    tx += ti * (x[i] - xm);
    ty += ti * (y[i] - ym);
  }
  return std::hypot(tx / tt, ty / tt);
}

void Track::update_motion(const std::array<double, 2>* det_vel) {
  const TrackerConfig& cfg = *cfg_;
  const double on = cfg.motion_on_speed[label], off = cfg.motion_off_speed[label];
  const std::optional<double> traj = displacement_speed();
  const double traj_speed = traj ? *traj : 0.0;
  const double det_speed = det_vel ? std::hypot((*det_vel)[0], (*det_vel)[1]) : traj_speed;
  const bool fast = traj && traj_speed > on && det_speed > on;
  speeds_ = {traj ? traj_speed : kNaN, det_speed};
  fast_.push_back(fast);
  while (static_cast<int>(fast_.size()) > cfg.motion_window) fast_.pop_front();
  if (fast) last_fast_ = stamp;
  const int n_fast = static_cast<int>(std::count(fast_.begin(), fast_.end(), true));
  if (n_fast >= cfg.motion_on_frames) {
    motion = Motion::Moving;
  } else if (motion == Motion::Moving && stamp - *last_fast_ >= cfg.motion_off_time && traj_speed < off) {
    motion = Motion::Stopped;
  }
  if (motion == Motion::Moving) {
    last_moving_ = stamp;
  } else if (motion == Motion::Stopped && stamp - *last_moving_ >= cfg.stop_memory) {
    motion = Motion::Static;
  }
}

void Track::update_yaw(double yaw_det) {
  const StateVec x = imm.mean();
  const double speed = std::hypot(x[2], x[3]);
  if (std::string(kClasses[label]) == "pedestrian") {
    // pedestrian boxes are about square: their heading only means something when walking
    if (speed > 0.4) yaw = std::atan2(x[3], x[2]);
    return;
  }
  if (std::abs(wrap_angle(yaw_det - yaw)) > kPi / 2) yaw_det += kPi;   // the 180 deg ambiguity
  if (speed > cfg_->heading_from_velocity) {
    const double heading = std::atan2(x[3], x[2]);
    if (std::abs(wrap_angle(yaw_det - heading)) > kPi / 2) yaw_det += kPi;
  }
  yaw = wrap_angle(yaw + cfg_->yaw_alpha * wrap_angle(yaw_det - yaw));
}

void Track::mark_missed() {
  score *= 0.9;
  record(false);
}

void Track::record(bool observed) {
  const StateVec x = imm.mean();
  history_.push_back({stamp, x[0], x[1], x[2], x[3], observed ? 1.0 : 0.0});
  while (!history_.empty() && history_.front()[0] < stamp - cfg_->history) history_.pop_front();
}

TrackState Track::to_state() const {
  StateVec x;
  StateMat P;
  imm.state(x, P);
  TrackState s;
  s.track_id = id;
  s.label = label;
  s.stamp = stamp;
  s.position = Vec3(x[0], x[1], z);
  s.velocity = Vec2(x[2], x[3]);
  s.yaw = yaw;
  s.size = size;
  s.score = score;
  s.covariance = P.topLeftCorner<4, 4>();
  s.mode_probs = imm.mu();
  s.age = stamp - birth;
  s.hits = hits;
  s.coasting = time_since_update() > 1e-6;
  s.history.assign(history_.begin(), history_.end());
  s.motion = motion;
  s.time_since_moving = last_moving_ ? stamp - *last_moving_ : kInf;
  s.speed_displacement = speeds_[0];
  s.speed_head = speeds_[1];
  s.has_measurement = !s.coasting;
  if (s.has_measurement) s.measurement = meas_;
  return s;
}

MultiObjectTracker::MultiObjectTracker(TrackerConfig cfg) : cfg_(std::make_shared<const TrackerConfig>(std::move(cfg))) {}

void MultiObjectTracker::reset() {
  tracks_.clear();
  next_id_ = 1;
  last_matches_.clear();
}

Matches MultiObjectTracker::associate(const Detections& dets, const std::vector<int>& det_idx,
                                      const std::vector<int>& trk_idx) const {
  if (det_idx.empty() || trk_idx.empty()) return {};
  Eigen::MatrixXd cost(det_idx.size(), trk_idx.size());
  std::vector<StateVec> tx;
  std::vector<double> gates;
  for (int t : trk_idx) {
    tx.push_back(tracks_[t].imm.mean());
    gates.push_back(tracks_[t].gate());
  }
  for (size_t a = 0; a < det_idx.size(); ++a)
    for (size_t b = 0; b < trk_idx.size(); ++b) {
      const Box& bx = dets.boxes[det_idx[a]];
      const double dx = bx[0] - tx[b][0], dy = bx[1] - tx[b][1];
      const double d = std::sqrt(dx * dx + dy * dy);
      const bool same = dets.labels[det_idx[a]] == tracks_[trk_idx[b]].label;
      cost(a, b) = same && d <= gates[b] ? d : kInf;
    }
  Matches m = match(cost, 1e9, cfg_->matching == "hungarian");
  for (auto& [d, t] : m) {
    d = det_idx[d];
    t = trk_idx[t];
  }
  return m;
}

std::vector<TrackState> MultiObjectTracker::step(const Detections& dets, double st) {
  const TrackerConfig& cfg = *cfg_;
  for (auto& t : tracks_) t.predict(st);

  std::vector<int> high, low;
  for (size_t i = 0; i < dets.size(); ++i) {
    const int l = dets.labels[i];
    if (std::find(cfg.classes.begin(), cfg.classes.end(), l) == cfg.classes.end()) continue;
    const double thr = cfg.high_score[l];
    if (dets.scores[i] >= thr) high.push_back(static_cast<int>(i));
    else if (dets.scores[i] >= cfg.low_score) low.push_back(static_cast<int>(i));
  }
  std::vector<int> all_trk(tracks_.size());
  std::iota(all_trk.begin(), all_trk.end(), 0);
  Matches matches = associate(dets, high, all_trk);
  std::vector<char> matched_t(tracks_.size(), 0);
  for (const auto& [d, t] : matches) matched_t[t] = 1;
  std::vector<int> rest;
  for (int i : all_trk)
    if (!matched_t[i] && tracks_[i].confirmed) rest.push_back(i);
  const Matches m2 = associate(dets, low, rest);
  matches.insert(matches.end(), m2.begin(), m2.end());

  last_matches_.clear();
  for (const auto& [d, t] : matches) {
    tracks_[t].update(dets.boxes[d], cfg.use_velocity ? &dets.velocities[d] : nullptr, dets.scores[d]);
    last_matches_.emplace_back(tracks_[t].id, d);
  }
  std::fill(matched_t.begin(), matched_t.end(), 0);
  std::vector<char> matched_d(dets.size(), 0);
  for (const auto& [d, t] : matches) {
    matched_t[t] = 1;
    matched_d[d] = 1;
  }
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (!matched_t[i]) tracks_[i].mark_missed();

  std::vector<int> fresh;
  for (int d : high)
    if (!matched_d[d]) fresh.push_back(d);
  std::stable_sort(fresh.begin(), fresh.end(), [&](int a, int b) { return dets.scores[a] > dets.scores[b]; });
  for (int d : fresh) {
    const Box& box = dets.boxes[d];
    const int l = dets.labels[d];
    bool second = false;
    for (const auto& t : tracks_)      // the tracks started in this loop count too
      if (second_box_on_track(box, l, t)) {
        second = true;
        break;
      }
    if (second) continue;
    tracks_.emplace_back(next_id_, l, box, cfg.use_velocity ? &dets.velocities[d] : nullptr, dets.scores[d], st,
                         cfg_);
    last_matches_.emplace_back(next_id_, d);
    ++next_id_;
  }

  std::vector<Track> kept;
  for (auto& t : tracks_) {
    const double tsu = t.time_since_update();
    if ((t.confirmed && tsu <= cfg.class_params[t.label].max_age) || (!t.confirmed && tsu <= cfg.tentative_max_age))
      kept.push_back(std::move(t));
  }
  tracks_ = std::move(kept);
  return outputs();
}

std::vector<const Track*> MultiObjectTracker::reported_tracks() const {
  std::vector<const Track*> out;
  for (const auto& t : tracks_) {
    if (!t.confirmed) continue;
    if (t.time_since_update() > 1e-6 && !cfg_->output_coasting) continue;
    if (t.score < cfg_->output_min_score[t.label]) continue;
    out.push_back(&t);
  }
  return out;
}

std::vector<TrackState> MultiObjectTracker::outputs() const {
  std::vector<TrackState> out;
  for (const Track* t : reported_tracks()) out.push_back(t->to_state());
  return out;
}

}  // namespace lidar3d
