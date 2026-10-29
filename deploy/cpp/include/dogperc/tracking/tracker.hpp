// Multi-object tracker in the gravity-aligned world frame (tracking/tracker.py).
//
// Association: high-score detections against all tracks, low-score detections against the still
// unmatched confirmed tracks, unmatched high-score detections start tentative tracks unless they are a
// second box on a track already kept. Life cycle, motion state (static / moving / stopped) and the
// smoothing are those of the Python tracker, step by step.
#pragma once

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dogperc/config.hpp"
#include "dogperc/detections.hpp"
#include "dogperc/tracking/association.hpp"
#include "dogperc/tracking/imm.hpp"

namespace dogperc {

enum class Motion { Static = 0, Moving = 1, Stopped = 2 };
const char* motion_name(Motion m);

struct TrackState {
  int track_id = 0;
  int label = 0;
  double stamp = 0.0;
  Vec3 position = Vec3::Zero();          // world
  Vec2 velocity = Vec2::Zero();
  double yaw = 0.0;
  Vec3 size = Vec3::Zero();              // l, w, h
  double score = 0.0;
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();   // [px, py, vx, vy]
  std::vector<double> mode_probs;
  double age = 0.0;
  int hits = 0;
  bool coasting = false;
  std::vector<std::array<double, 6>> history;   // [t, x, y, vx, vy, observed]
  Motion motion = Motion::Static;
  double time_since_moving = kInf;
  double speed_displacement = kNaN;
  double speed_head = kNaN;
  bool has_measurement = false;
  std::array<double, 6> measurement{};          // matched detection: x, y, yaw, vx, vy, score
  const char* name() const { return kClasses[label]; }
};

class Track {
 public:
  // cfg: shared with the tracker (immutable), so tracks and trackers can be copied and moved freely
  Track(int id, int label, const Box& box, const std::array<double, 2>* vel, double score, double stamp,
        std::shared_ptr<const TrackerConfig> cfg);
  void predict(double stamp);
  double gate() const;
  void update(const Box& box, const std::array<double, 2>* vel, double score);
  void mark_missed();
  std::optional<double> displacement_speed() const;
  TrackState to_state() const;

  int id;
  int label;
  IMM imm;
  double z, yaw, score;
  Vec3 size;
  double birth, last_update, stamp;
  int hits;
  bool confirmed;
  Motion motion = Motion::Static;
  double time_since_update() const { return stamp - last_update; }
  const ClassParams& params() const { return cfg_->class_params[label]; }

 private:
  void measure(const Box& box, const std::array<double, 2>* vel, double score);
  void add_center(const Box& box);
  void update_motion(const std::array<double, 2>* det_vel);
  void update_yaw(double yaw_det);
  void record(bool observed);

  std::shared_ptr<const TrackerConfig> cfg_;
  std::deque<std::array<double, 6>> history_;
  std::optional<double> last_moving_, last_fast_;
  std::deque<bool> fast_;
  std::deque<std::array<double, 3>> centers_;   // (stamp, x, y) of the detections
  std::array<double, 6> meas_{};
  std::array<double, 2> speeds_{kNaN, kNaN};
};

class MultiObjectTracker {
 public:
  explicit MultiObjectTracker(TrackerConfig cfg = {});
  void reset();
  // dets in the world frame; returns the reported tracks
  std::vector<TrackState> step(const Detections& dets, double stamp);
  std::vector<const Track*> reported_tracks() const;
  std::vector<TrackState> outputs() const;
  const std::vector<Track>& tracks() const { return tracks_; }
  // track id -> detection index of the last step (matched or started)
  const std::vector<std::pair<int, int>>& last_matches() const { return last_matches_; }
  const TrackerConfig& config() const { return *cfg_; }

 private:
  Matches associate(const Detections& dets, const std::vector<int>& det_idx, const std::vector<int>& trk_idx) const;

  std::shared_ptr<const TrackerConfig> cfg_;
  std::vector<Track> tracks_;
  int next_id_ = 1;
  std::vector<std::pair<int, int>> last_matches_;
};

}  // namespace dogperc
