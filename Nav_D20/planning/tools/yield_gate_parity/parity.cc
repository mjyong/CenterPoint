// Frame-by-frame comparison of the C++ yield gate (src/algorithm/yield_gate/) with the Python reference, on the
// golden file make_golden.py writes. Exit 1 on any difference.
//
//   bash Nav_D20/planning/tools/yield_gate_parity/run.sh            (builds, makes the golden data, compares)
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "src/algorithm/yield_gate/yield_gate.h"

using namespace planning::algorithm::yield;

namespace {

std::ifstream in;

std::string Tok() {
  std::string t;
  if (!(in >> t)) throw std::runtime_error("unexpected end of the golden file");
  return t;
}
double Num() { return std::stod(Tok()); }
int Int() { return std::stoi(Tok()); }
// two numbers in file order (the order of evaluation of function arguments is unspecified)
Vec2 Xy() {
  const double x = Num();
  const double y = Num();
  return Vec2(x, y);
}

bool Close(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::abs(a - b) <= 1e-6 * (1.0 + std::abs(a));
}

YieldConfig ReadConfig() {
  YieldConfig c;
  const int n = Int();
  for (int i = 0; i < n; ++i) {
    const std::string k = Tok();
    if (k == "classes") {
      c.classes.clear();
      const int m = Int();
      for (int j = 0; j < m; ++j) c.classes.push_back(Tok());
    } else if (k == "yield_rule") {
      c.yield_rule = Tok();
    } else {
      const double v = Num();
#define F(name) else if (k == #name) c.name = v
      if (false) {}
      F(dog_speed); F(dog_length); F(dog_width); F(lateral_margin); F(vehicle_margin); F(sigma_k);
      F(min_mode_prob); F(horizon); F(sigma_growth); F(sigma_max); F(dt); F(path_step); F(path_length); F(pet);
      F(clear_time); F(stopped_speed); F(stopped_wait); F(blocked_wait); F(start_zone); F(stale_after);
      F(young_age); F(young_range); F(turn_range); F(turn_radius); F(turn_max_deg); F(reaction_time);
      F(brake_decel); F(stop_margin); F(wait_reach); F(wait_max_horizon); F(wait_margin); F(wait_clear);
      F(lost_hold); F(min_watch);
      else throw std::runtime_error("unknown config key " + k);
#undef F
    }
  }
  return c;
}

Object ReadObject() {
  if (Tok() != "OBJ") throw std::runtime_error("expected OBJ");
  Object o;
  o.id = Int();
  o.label = Tok();
  const int m = Int();
  o.motion = m == 1 ? Motion::kMoving : m == 2 ? Motion::kStopped : Motion::kStatic;
  o.position = Xy();
  o.velocity = Xy();
  o.yaw = Num();
  o.length = Num();
  o.width = Num();
  const double age = Num(), sf = Num();
  o.has_age = !std::isnan(age);
  o.age = o.has_age ? age : 0.0;
  o.has_standing_for = !std::isnan(sf);
  o.standing_for = o.has_standing_for ? sf : 0.0;
  o.has_covariance = Int() == 1;
  if (o.has_covariance) {
    const double a = Num(), b = Num(), c = Num(), d = Num();
    o.covariance << a, b, c, d;
  }
  const int T = Int();
  for (int i = 0; i < T; ++i) o.pred_times.push_back(Num());
  if (T > 0) {
    const int K = Int();
    for (int k = 0; k < K; ++k) {
      PredictionMode pm;
      pm.probability = Num();
      for (int i = 0; i < T; ++i) pm.positions.push_back(Xy());
      if (Int() == 1) {
        for (int i = 0; i < T; ++i) {
          Mat2 c;
          const double a = Num(), b = Num(), cc = Num(), d = Num();
          c << a, b, cc, d;
          pm.covariances.push_back(c);
        }
      }
      o.modes.push_back(pm);
    }
  }
  return o;
}

std::string Ids(const std::vector<int> &v) {
  std::ostringstream s;
  for (int i : v) s << i << " ";
  return s.str();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: parity <golden.txt>\n";
    return 2;
  }
  in.open(argv[1]);
  if (!in) {
    std::cerr << "cannot read " << argv[1] << "\n";
    return 2;
  }
  YieldConfig cfg;
  YieldGate gate(cfg);
  long frames = 0, bad = 0;
  std::string name;
  for (;;) {
    const std::string t = Tok();
    if (t == "END") break;
    if (t == "CONFIG") {
      cfg = ReadConfig();
      continue;
    }
    if (t == "RESET") {
      name = Tok();
      gate = YieldGate(cfg);
      continue;
    }
    if (t != "FRAME") throw std::runtime_error("unexpected token " + t);
    Tok();  // name
    const double now = Num(), stamp = Num();
    const bool moving = Int() == 1, cw = Int() == 1;
    std::vector<Vec2> path(Int());
    for (auto &p : path) p = Xy();
    std::vector<Object> objs(Int());
    for (auto &o : objs) o = ReadObject();
    // expected
    if (Tok() != "ASSESS") throw std::runtime_error("expected ASSESS");
    const bool e_stale = Int(), e_lane = Int();
    const double e_min_gap = Num();
    std::vector<Conflict> e_conf(Int());
    for (auto &c : e_conf) {
      c.id = Int();
      c.gap = Num();
      c.s = Num();
      c.kind = Tok();
    }
    std::vector<Standing> e_st(Int());
    for (auto &st : e_st) {
      st.id = Int();
      st.standing_for = Num();
      st.has_standing_for = !std::isnan(st.standing_for);
      st.blocking = Int();
    }
    std::set<int> e_app, e_pres;
    for (int i = 0, n = Int(); i < n; ++i) e_app.insert(Int());
    for (int i = 0, n = Int(); i < n; ++i) e_pres.insert(Int());
    if (Tok() != "DECISION") throw std::runtime_error("expected DECISION");
    Decision e_d;
    e_d.go = Int();
    e_d.reason = Tok();
    for (int i = 0, n = Int(); i < n; ++i) e_d.ids.push_back(Int());
    e_d.gap = Num();

    const Assessment a = Assess(path, objs, now, stamp, cfg);
    const Decision d = gate.Update(a, now, moving, cw);
    ++frames;
    std::ostringstream why;
    if (a.stale != e_stale) why << " stale " << a.stale << "/" << e_stale;
    if (a.in_lane != e_lane) why << " in_lane " << a.in_lane << "/" << e_lane;
    if (!Close(a.min_gap, e_min_gap)) why << " min_gap " << a.min_gap << "/" << e_min_gap;
    if (a.conflicts.size() != e_conf.size()) {
      why << " conflicts " << a.conflicts.size() << "/" << e_conf.size();
    } else {
      for (size_t i = 0; i < e_conf.size(); ++i) {
        const auto &x = a.conflicts[i], &y = e_conf[i];
        if (x.id != y.id || x.kind != y.kind || !Close(x.gap, y.gap) || !Close(x.s, y.s)) {
          why << " conflict[" << i << "] " << x.id << " " << x.kind << " " << x.gap << " " << x.s << " / " << y.id
              << " " << y.kind << " " << y.gap << " " << y.s;
        }
      }
    }
    if (a.standing.size() != e_st.size()) {
      why << " standing " << a.standing.size() << "/" << e_st.size();
    } else {
      for (size_t i = 0; i < e_st.size(); ++i) {
        const auto &x = a.standing[i], &y = e_st[i];
        const double xs = x.has_standing_for ? x.standing_for : NAN;
        if (x.id != y.id || x.blocking != y.blocking || !Close(xs, y.standing_for)) {
          why << " standing[" << i << "] " << x.id << " " << xs << " " << x.blocking << " / " << y.id << " "
              << y.standing_for << " " << y.blocking;
        }
      }
    }
    if (a.approaching != e_app) why << " approaching";
    if (a.present != e_pres) why << " present";
    if (d.go != e_d.go || d.reason != e_d.reason || d.ids != e_d.ids || !Close(d.gap, e_d.gap)) {
      why << " decision " << d.go << " " << d.reason << " [" << Ids(d.ids) << "] " << d.gap << " / " << e_d.go << " "
          << e_d.reason << " [" << Ids(e_d.ids) << "] " << e_d.gap;
    }
    if (!why.str().empty()) {
      if (++bad <= 30) std::cout << "DIFF " << name << " frame " << frames << " t " << now << ":" << why.str() << "\n";
    }
  }
  std::cout << frames << " frames, " << bad << " differ\n";
  return bad == 0 ? 0 : 1;
}
