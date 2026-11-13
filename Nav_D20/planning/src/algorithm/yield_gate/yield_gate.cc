// C++ port of dog_perception/yield_gate.py; see yield_gate.h. Kept line by line with the Python (numpy's interp,
// arange and argmin semantics) so the two agree frame by frame (tools/yield_gate_parity/).
#include "src/algorithm/yield_gate/yield_gate.h"

#include <algorithm>

namespace planning {
namespace algorithm {
namespace yield {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// np.arange(0, stop + 1e-9, step)
std::vector<double> Arange(double stop, double step) {
  const int n = std::max(0, static_cast<int>(std::ceil((stop + 1e-9) / step)));
  std::vector<double> out(n);
  for (int i = 0; i < n; ++i) out[i] = i * step;
  return out;
}

// np.interp(x, xp, fp) for increasing xp
double Interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
  const size_t n = xp.size();
  if (n == 1 || x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  const size_t j = static_cast<size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
  const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  double r = slope * (x - xp[j]) + fp[j];
  if (std::isnan(r)) {
    r = slope * (x - xp[j + 1]) + fp[j + 1];
    if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
  }
  return r;
}

double LamMax(const Mat2 &c) {
  const double a = c(0, 0), b = c(0, 1), d = c(1, 1);
  return 0.5 * (a + d) + std::sqrt(std::max(0.25 * (a - d) * (a - d) + b * b, 0.0));
}

struct ModeTrack {
  std::vector<Vec2> centers;
  std::vector<double> heads;
  std::vector<double> sigma;
};

bool IsMoving(const Object &o, const YieldConfig &cfg) {
  return o.motion == Motion::kMoving || std::hypot(o.velocity.x(), o.velocity.y()) >= cfg.stopped_speed;
}

// First and last time each point is inside the oriented box (centers[k], heads[k], half_l[k], half_w[k]) at
// times[k]; NaN where never.
void Occupancy(const std::vector<Vec2> &pts, const std::vector<Vec2> &centers, const std::vector<double> &heads,
               const std::vector<double> &half_l, const std::vector<double> &half_w,
               const std::vector<double> &times, std::vector<double> *first, std::vector<double> *last) {
  const size_t J = pts.size(), K = centers.size();
  first->assign(J, kNaN);
  last->assign(J, kNaN);
  std::vector<double> c(K), s(K);
  for (size_t k = 0; k < K; ++k) {
    c[k] = std::cos(heads[k]);
    s[k] = std::sin(heads[k]);
  }
  for (size_t j = 0; j < J; ++j) {
    double f = kInf, l = -kInf;
    bool any = false;
    for (size_t k = 0; k < K; ++k) {
      const double dx = pts[j].x() - centers[k].x(), dy = pts[j].y() - centers[k].y();
      if (std::abs(dx * c[k] + dy * s[k]) <= half_l[k] && std::abs(-dx * s[k] + dy * c[k]) <= half_w[k]) {
        any = true;
        f = std::min(f, times[k]);
        l = std::max(l, times[k]);
      }
    }
    if (any) {
      (*first)[j] = f;
      (*last)[j] = l;
    }
  }
}

std::vector<ModeTrack> Rollout(const Object &o, double now, double stamp, const YieldConfig &cfg, double horizon,
                               std::vector<double> *tau_out) {
  const std::vector<double> tau = Arange(horizon, cfg.dt);
  std::vector<double> q(tau.size());
  for (size_t i = 0; i < tau.size(); ++i) q[i] = tau[i] + std::max(0.0, now - stamp);
  const Vec2 p0 = o.position, v0 = o.velocity;
  const Mat2 cov0 = o.has_covariance ? o.covariance : Mat2::Zero();
  const size_t T = o.pred_times.size();
  // modes: a single constant-velocity mode without a prediction
  std::vector<double> probs;
  if (T == 0 || o.modes.empty()) {
    probs = {1.0};
  } else {
    for (const auto &m : o.modes) probs.push_back(m.probability);
  }
  std::vector<size_t> order(probs.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
  std::vector<size_t> keep{order[0]};
  for (size_t i = 1; i < order.size(); ++i) {
    if (probs[order[i]] >= cfg.min_mode_prob) keep.push_back(order[i]);
  }
  const bool has_pred = T > 0 && !o.modes.empty();
  std::vector<ModeTrack> out;
  for (size_t k : keep) {
    std::vector<double> t{0.0}, mx{p0.x()}, my{p0.y()}, lam{LamMax(cov0)};
    if (has_pred) {
      const PredictionMode &m = o.modes[k];
      const bool covs = !m.covariances.empty();
      for (size_t i = 0; i < T; ++i) {
        t.push_back(o.pred_times[i]);
        mx.push_back(m.positions[i].x());
        my.push_back(m.positions[i].y());
        lam.push_back(covs ? LamMax(m.covariances[i]) : LamMax(cov0));
      }
    }
    ModeTrack mt;
    const size_t n = q.size();
    std::vector<double> x(n), y(n), var(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = Interp(q[i], t, mx);
      y[i] = Interp(q[i], t, my);
      var[i] = Interp(q[i], t, lam);
    }
    Vec2 v_end = v0;
    if (has_pred) {
      const size_t e = t.size() - 1;
      v_end = Vec2(mx[e] - mx[e - 1], my[e] - my[e - 1]) / std::max(t[e] - t[e - 1], 1e-6);
    }
    const double t_last = t.back();
    for (size_t i = 0; i < n; ++i) {
      if (q[i] > t_last) {
        x[i] = mx.back() + v_end.x() * (q[i] - t_last);
        y[i] = my.back() + v_end.y() * (q[i] - t_last);
        var[i] = var[i] + std::pow(cfg.sigma_growth * (q[i] - t_last), 2);
      }
      var[i] = std::min(var[i], cfg.sigma_max * cfg.sigma_max);
    }
    mt.centers.resize(n);
    for (size_t i = 0; i < n; ++i) mt.centers[i] = Vec2(x[i], y[i]);
    mt.heads.resize(n);
    if (n >= 2) {
      std::vector<Vec2> step(n - 1);
      for (size_t i = 0; i + 1 < n; ++i) step[i] = mt.centers[i + 1] - mt.centers[i];
      for (size_t i = 0; i + 1 < n; ++i) mt.heads[i] = std::atan2(step[i].y(), step[i].x());
      mt.heads[n - 1] = mt.heads[n - 2];
      for (size_t i = 0; i < n; ++i) {
        const Vec2 &st = i + 1 < n ? step[i] : step[n - 2];
        if (st.norm() < 1e-3) mt.heads[i] = o.yaw;
      }
    } else if (n == 1) {
      mt.heads[0] = o.yaw;   // np.array([yaw]), and still (no step) -> yaw
    }
    mt.sigma.resize(n);
    for (size_t i = 0; i < n; ++i) mt.sigma[i] = std::sqrt(var[i]);
    out.push_back(std::move(mt));
  }
  *tau_out = tau;
  return out;
}

std::vector<ModeTrack> TurnModes(const Object &o, double now, double stamp, const YieldConfig &cfg, double horizon,
                                 const std::vector<double> &sigma) {
  const std::vector<double> tau = Arange(horizon, cfg.dt);
  const Vec2 p0 = o.position, v0 = o.velocity;
  const double speed = std::hypot(v0.x(), v0.y());
  const double th0 = std::atan2(v0.y(), v0.x());
  std::vector<ModeTrack> out;
  for (double sign : {1.0, -1.0}) {
    const double w = sign * speed / cfg.turn_radius;
    const double t_turn = (cfg.turn_max_deg * M_PI / 180.0) / std::abs(w);
    ModeTrack mt;
    mt.sigma = sigma;
    for (size_t i = 0; i < tau.size(); ++i) {
      const double q = tau[i] + std::max(0.0, now - stamp);
      const double qt = std::min(q, t_turn);
      const double th = th0 + w * qt;
      const double x = p0.x() + speed / w * (std::sin(th) - std::sin(th0));
      const double y = p0.y() - speed / w * (std::cos(th) - std::cos(th0));
      const double rest = std::max(q - t_turn, 0.0);
      mt.centers.emplace_back(x + speed * rest * std::cos(th), y + speed * rest * std::sin(th));
      mt.heads.push_back(th);
    }
    out.push_back(std::move(mt));
  }
  return out;
}

}  // namespace

void ResamplePath(const std::vector<Vec2> &path, double step, double length, std::vector<Vec2> *pts,
                  std::vector<double> *s) {
  pts->clear();
  s->clear();
  if (path.empty()) return;
  std::vector<double> cum{0.0}, px{path[0].x()}, py{path[0].y()};
  for (size_t i = 1; i < path.size(); ++i) {
    const double d = (path[i] - path[i - 1]).norm();
    if (d <= 0.0) continue;   // repeated points (np.interp needs increasing s)
    cum.push_back(cum.back() + d);
    px.push_back(path[i].x());
    py.push_back(path[i].y());
  }
  if (cum.size() == 1) {
    pts->push_back(path[0]);
    s->push_back(0.0);
    return;
  }
  *s = Arange(std::min(cum.back(), length), step);
  for (double v : *s) pts->emplace_back(Interp(v, cum, px), Interp(v, cum, py));
}

bool InsidePolygon(const Vec2 &p, const std::vector<Vec2> &poly) {
  int crossings = 0;
  const size_t n = poly.size();
  for (size_t i = 0; i < n; ++i) {
    const Vec2 &a = poly[i], &b = poly[(i + 1) % n];
    if ((a.y() > p.y()) != (b.y() > p.y())) {
      const double xc = a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y());
      if (p.x() < xc) ++crossings;
    }
  }
  return crossings % 2 == 1;
}

bool CrosswalkSpan(const std::vector<Vec2> &path, const std::vector<Vec2> &polygon, double *s_in, double *s_out,
                   double step) {
  std::vector<Vec2> pts;
  std::vector<double> s;
  ResamplePath(path, step, kInf, &pts, &s);
  size_t i = 0;
  while (i < pts.size() && !InsidePolygon(pts[i], polygon)) ++i;
  if (i == pts.size()) return false;
  size_t j = i;
  while (j + 1 < pts.size() && InsidePolygon(pts[j + 1], polygon)) ++j;
  *s_in = s[i];
  *s_out = s[j];
  return true;
}

Assessment Assess(const std::vector<Vec2> &path, const std::vector<Object> &objects, double now, double stamp,
                  const YieldConfig &cfg) {
  Assessment a;
  a.now = now;
  a.stale = std::isnan(stamp) || now - stamp > cfg.stale_after;
  if (a.stale) return a;
  std::vector<Vec2> pts;
  std::vector<double> s;
  ResamplePath(path, cfg.path_step, cfg.path_length, &pts, &s);
  const size_t J = pts.size();
  std::vector<double> d_in(J), d_out(J);
  for (size_t j = 0; j < J; ++j) {
    d_in[j] = std::max(0.0, s[j] - 0.5 * cfg.dog_length) / cfg.dog_speed;
    d_out[j] = (s[j] + 0.5 * cfg.dog_length) / cfg.dog_speed;
  }
  const double horizon = std::max(cfg.horizon, d_out.back() + cfg.pet);
  const double corridor = 0.5 * cfg.dog_width + cfg.lateral_margin;
  std::vector<Vec2> here;
  for (size_t j = 0; j < J; ++j) {
    if (s[j] <= 0.5 * cfg.dog_length) here.push_back(pts[j]);
  }
  const bool wait = cfg.yield_rule == "wait";
  const bool stop = cfg.yield_rule == "stop";
  for (const Object &o : objects) {
    if (std::find(cfg.classes.begin(), cfg.classes.end(), o.label) == cfg.classes.end()) continue;
    a.present.insert(o.id);
    const double hl0 = 0.5 * o.length, hw0 = 0.5 * o.width;
    std::vector<double> dist(J);
    size_t jn = 0;
    for (size_t j = 0; j < J; ++j) {
      dist[j] = (pts[j] - o.position).norm();
      if (dist[j] < dist[jn]) jn = j;
    }
    bool conflicted = false;
    if (IsMoving(o, cfg)) {
      const double speed = std::hypot(o.velocity.x(), o.velocity.y());
      double h = horizon;
      if (wait) {
        h = std::min(std::max(cfg.wait_reach / std::max(speed, 1e-3), horizon),
                     std::max(horizon, cfg.wait_max_horizon));
      }
      std::vector<double> tau;
      std::vector<ModeTrack> modes = Rollout(o, now, stamp, cfg, h, &tau);
      std::vector<std::string> kinds(modes.size(), "predicted");
      if ((wait || cfg.turn_range > 0) && (wait || dist[jn] <= cfg.turn_range)) {
        for (auto &m : TurnModes(o, now, stamp, cfg, h, modes[0].sigma)) {
          modes.push_back(std::move(m));
          kinds.push_back("turn");
        }
      }
      double worst_g = kInf, worst_s = kNaN;
      std::string worst_kind;
      const double need = cfg.reaction_time + speed / (2.0 * cfg.brake_decel) + cfg.stop_margin;
      for (size_t m = 0; m < modes.size(); ++m) {
        const ModeTrack &mt = modes[m];
        const size_t K = mt.centers.size();
        std::vector<double> hl(K), hw(K);
        for (size_t k = 0; k < K; ++k) {
          const double grow =
              cfg.vehicle_margin + cfg.sigma_k * mt.sigma[k] + corridor + (wait ? cfg.wait_margin : 0.0);
          hl[k] = hl0 + grow;
          hw[k] = hw0 + grow;
        }
        std::vector<double> first, last;
        Occupancy(pts, mt.centers, mt.heads, hl, hw, tau, &first, &last);
        size_t jm = 0;
        double gm = kInf;
        for (size_t j = 0; j < J; ++j) {
          double g;
          if (wait) {
            g = std::isnan(first[j]) ? kInf : first[j];
          } else if (stop) {
            g = (std::isnan(first[j]) ? kInf : first[j]) - need;
          } else {
            g = std::isnan(first[j]) ? kInf : std::max(first[j] - d_out[j], d_in[j] - last[j]);
          }
          if (j == 0 || g < gm) {   // np.argmin: the first of the smallest
            gm = g;
            jm = j;
          }
        }
        if (gm < worst_g) {
          worst_g = gm;
          worst_s = s[jm];
          worst_kind = kinds[m];
        }
        if (!a.in_lane && kinds[m] == "predicted") {
          std::vector<double> rl(K, hl0), rw(K, hw0 + 0.5 * cfg.dog_width), rf, rlast;
          Occupancy(here, mt.centers, mt.heads, rl, rw, tau, &rf, &rlast);
          for (double v : rf) {
            if (!std::isnan(v)) a.in_lane = true;
          }
        }
      }
      a.min_gap = std::min(a.min_gap, worst_g);
      const double limit = wait ? kInf : stop ? 0.0 : cfg.pet;
      if (worst_g < limit) {
        a.conflicts.push_back(Conflict{o.id, worst_g, worst_s, worst_kind});
        conflicted = true;
      }
      const Vec2 r = pts[jn] - o.position;
      if (o.velocity.dot(r) > 0) a.approaching.insert(o.id);
    } else {
      // standing: in the corridor, or nose / tail towards it within start_zone
      const double c = std::cos(o.yaw), sn = std::sin(o.yaw);
      bool near_any = false, blocking = false;
      for (size_t j = 0; j < J; ++j) {
        const Vec2 d = pts[j] - o.position;
        const double along = d.x() * c + d.y() * sn;
        const double across = -d.x() * sn + d.y() * c;
        if (std::abs(across) <= hw0 + cfg.vehicle_margin + corridor && std::abs(along) <= hl0 + cfg.start_zone) {
          near_any = true;
        }
        if (std::abs(across) <= hw0 + corridor && std::abs(along) <= hl0 + corridor) blocking = true;
      }
      if (near_any) {
        Standing st{o.id, o.has_standing_for, o.standing_for, blocking};
        if (!st.has_standing_for && o.motion == Motion::kStatic && o.has_age) {
          st.has_standing_for = true;
          st.standing_for = o.age;
        }
        a.standing.push_back(st);
      }
    }
    // a new track: its velocity (and so its prediction) is not to be trusted yet
    if (!conflicted && cfg.young_range > 0 && o.has_age && o.age < cfg.young_age && dist[jn] <= cfg.young_range) {
      a.conflicts.push_back(Conflict{o.id, kNaN, s[jn], "young"});
    }
  }
  return a;
}

void YieldGate::Reset() {
  paused_ = false;
  last_conflict_ = -kInf;
  yielded_.clear();
  still_since_.clear();
  conflict_last_.clear();
  seen_last_.clear();
  watch_set_ = false;
  watch_since_ = 0.0;
}

Decision YieldGate::Update(const Assessment &a, double now, bool dog_moving, bool on_crosswalk) {
  if (!watch_set_ && !dog_moving && !on_crosswalk) {
    watch_set_ = true;
    watch_since_ = now;
  }
  Decision d = Decide(a, now, dog_moving, on_crosswalk);
  if (d.go && d.reason == "clear" && cfg_.min_watch > 0 && !dog_moving && !on_crosswalk &&
      now - watch_since_ < cfg_.min_watch) {
    return Decision{false, "watching", {}, d.gap};
  }
  return d;
}

Decision YieldGate::Decide(const Assessment &a, double now, bool dog_moving, bool on_crosswalk) {
  const YieldConfig &cfg = cfg_;
  std::vector<int> conflict_ids;
  for (const auto &c : a.conflicts) conflict_ids.push_back(c.id);
  if (on_crosswalk) {
    paused_ = false;
    yielded_.clear();
    return Decision{true, "on_crosswalk", conflict_ids, a.min_gap};
  }
  if (a.stale) {
    paused_ = true;
    last_conflict_ = now;
    return Decision{false, "stale", {}, kInf};
  }
  std::set<int> standing_ids;
  for (const auto &st : a.standing) standing_ids.insert(st.id);
  for (auto it = still_since_.begin(); it != still_since_.end();) {
    it = standing_ids.count(it->first) ? std::next(it) : still_since_.erase(it);
  }
  std::vector<int> waiting_for;
  bool blocked = false;
  for (const auto &st : a.standing) {
    auto it = still_since_.find(st.id);
    if (it == still_since_.end()) {
      it = still_since_.emplace(st.id, st.has_standing_for ? now - st.standing_for : now).first;
    }
    if (now - it->second < (st.blocking ? cfg.blocked_wait : cfg.stopped_wait)) {
      waiting_for.push_back(st.id);
      blocked = blocked || st.blocking;
    }
  }
  double gap = kInf;
  bool any_gap = false;
  for (const auto &c : a.conflicts) {
    if (!std::isnan(c.gap)) {
      gap = any_gap ? std::min(gap, c.gap) : c.gap;
      any_gap = true;
    }
  }
  if (!any_gap) gap = a.min_gap;
  const bool wait = cfg.yield_rule == "wait";
  if (wait) {
    for (int oid : a.present) seen_last_[oid] = now;
    for (int oid : conflict_ids) conflict_last_[oid] = now;
  }
  if (dog_moving && !paused_ && a.in_lane) {
    return Decision{true, "committed", conflict_ids, gap};   // never stop in front of a vehicle
  }
  if (!conflict_ids.empty() || !waiting_for.empty()) {
    paused_ = true;
    last_conflict_ = now;
    std::vector<int> ids = conflict_ids;
    ids.insert(ids.end(), waiting_for.begin(), waiting_for.end());
    yielded_.insert(ids.begin(), ids.end());
    std::string reason;
    if (!conflict_ids.empty()) {
      const bool all_young = std::all_of(a.conflicts.begin(), a.conflicts.end(),
                                         [](const Conflict &c) { return c.kind == "young"; });
      reason = all_young ? "young" : "vehicle";
    } else {
      reason = blocked ? "blocked" : "standing";
    }
    return Decision{false, reason, ids, gap};
  }
  auto last_of = [](const std::map<int, double> &m, int id) {
    auto it = m.find(id);
    return it == m.end() ? -kInf : it->second;
  };
  if (paused_ && wait) {
    std::vector<int> clearing, lost;
    for (int i : yielded_) {
      if (a.present.count(i) && now - last_of(conflict_last_, i) < cfg.wait_clear) clearing.push_back(i);
    }
    if (!clearing.empty()) return Decision{false, "clearing", clearing, gap};
    for (int i : yielded_) {
      if (!a.present.count(i) && now - last_of(seen_last_, i) < cfg.lost_hold) lost.push_back(i);
    }
    if (!lost.empty()) return Decision{false, "lost", lost, gap};
    paused_ = false;
    yielded_.clear();
    return Decision{true, "clear", {}, gap};
  }
  if (paused_) {
    if (now - last_conflict_ < cfg.clear_time) {
      return Decision{false, "clearing", std::vector<int>(yielded_.begin(), yielded_.end()), gap};
    }
    std::vector<int> pending;
    for (int i : yielded_) {
      if (a.present.count(i) && a.approaching.count(i)) pending.push_back(i);
    }
    if (!pending.empty()) return Decision{false, "not_passed", pending, gap};
    paused_ = false;
    yielded_.clear();
  }
  return Decision{true, "clear", {}, gap};
}

}  // namespace yield
}  // namespace algorithm
}  // namespace planning
