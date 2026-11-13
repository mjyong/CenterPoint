// Closed-loop check of CrosswalkYield (src/runnable/mission/tracking/crosswalk_yield.cc): a straight path north
// through a crosswalk at y 10..18; the robot follows the path at 1 m/s times the speed scale, or holds and turns.
// Cases: a car passing east on y = 14 while the robot waits; no car; no perception; arriving turned 60 deg away;
// the pose jumping 1.2 m onto the crosswalk while the robot waits (with the car, and with no perception); starting
// 3 m onto the crosswalk with no perception (T52 review).
// Exit 1 when a case does not do what it should. Built and run by run.sh.
#include <cstdio>
#include <string>

#include "src/runnable/mission/tracking/crosswalk_yield.h"

using namespace planning::mission;
using planning::algorithm::yield::Object;
using planning::algorithm::yield::Vec2;

namespace {

struct Result {
  std::string modes;            // the sequence of modes
  bool waited_for_car = false;  // a hold with reason vehicle
  bool held_on_crosswalk = false;
  bool aligned = false;         // an align hold
  bool unreleased = false;       // an output with "on the crosswalk without a go"
  double go_time = -1, arrive_time = -1, reach_time = -1, cross_time = -1;
};

Result Run(const std::string &mode) {
  const bool jump = mode == "jump" || mode == "jumpstale", deep = mode == "deep";
  const bool car = mode == "car" || mode == "jump", stale = mode == "stale" || mode == "jumpstale" || deep;
  const bool turned = mode == "turned";
  CrosswalkYieldConfig cfg;
  cfg.enable = true;
  CrosswalkYield cy;
  cy.Configure(cfg);
  cy.SetCrosswalks({{Vec2(-3, 10), Vec2(3, 10), Vec2(3, 18), Vec2(-3, 18)}});
  const std::vector<Vec2> path{Vec2(0.3, 0), Vec2(0.3, 30)};
  Vec2 xy(0.3, deep ? 13.0 : 0.0);
  bool jumped = false;
  double yaw = turned ? M_PI / 2 + 1.05 : M_PI / 2 + 0.6;
  const double dt = 0.05, t0 = 1000.0;
  Result r;
  CrosswalkYieldOutput::Mode last = CrosswalkYieldOutput::kNone;
  for (int k = 0; k < 1500; ++k) {
    const double t = t0 + k * dt;
    if (jump && !jumped && t - t0 > 14.0) {   // waiting at y 9.1: localization puts it 0.3 m past the entry
      xy.y() += 1.2;
      jumped = true;
    }
    cy.AddOdom(t, xy, yaw);
    if (k % 2 == 0 && !stale) {   // perception: 10 Hz, 0.1 s old, in its own frame
      std::vector<Object> objs;
      const Vec2 car_map(-100.0 + 6.0 * (t - 0.1 - t0), 14.0);   // 6 m/s east, crosses x = 0 at 16.7 s
      if (car && std::abs(car_map.x() - xy.x()) < 40) {
        Eigen::Matrix2d Rm, Rd;
        Rm << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
        Rd << 0, 1, -1, 0;   // det -> base_link: -90 deg
        const Vec2 pb = Rm.transpose() * (car_map - xy);
        Object o;
        o.id = 7;
        o.label = "vehicle";
        o.motion = planning::algorithm::yield::Motion::kMoving;
        o.position = Rd.transpose() * (pb - Vec2(0.319, 0.043));
        o.velocity = Rd.transpose() * Rm.transpose() * Vec2(6.0, 0.0);
        o.yaw = std::atan2(o.velocity.y(), o.velocity.x());
        o.length = 4.5;
        o.width = 1.9;
        o.has_age = true;
        o.age = 5.0;
        objs.push_back(o);
      }
      cy.SetObjects(t - 0.1, objs);
    }
    const CrosswalkYieldOutput out = cy.Step(t, xy, yaw, 0.0, path);
    if (out.mode != last) {
      std::printf("  %-7s t %5.2f  y %5.2f  yaw %5.1f deg  %-8s %-12s s_stop %5.2f\n", mode.c_str(), t - t0, xy.y(),
                  yaw * 180 / M_PI, ModeName(out.mode), out.decision.reason.c_str(), out.s_stop);
      r.modes += std::string(ModeName(out.mode)) + " ";
      last = out.mode;
    }
    if (out.mode == CrosswalkYieldOutput::kAlign && out.hold) r.aligned = true;
    if (out.hold && out.decision.reason == "vehicle") r.waited_for_car = true;
    if (r.arrive_time < 0 && out.hold) r.arrive_time = t - t0;
    if (r.go_time < 0 && out.mode == CrosswalkYieldOutput::kGo) r.go_time = t - t0;
    if (r.cross_time < 0 && out.mode == CrosswalkYieldOutput::kCross) r.cross_time = t - t0;
    if (out.unreleased) r.unreleased = true;
    if (out.hold && xy.y() > 10 && xy.y() < 18) r.held_on_crosswalk = true;
    double v = out.speed_scale;
    if (out.hold) {
      v = 0.0;
      yaw += out.hold_omega * dt;
    } else if (!turned || xy.y() > 9.5) {
      yaw += std::clamp(M_PI / 2 - yaw, -0.5 * dt, 0.5 * dt);   // the local planner turns along the path
    }
    xy.y() += v * dt;
    if (xy.y() > 25) {
      r.reach_time = t - t0;
      break;
    }
  }
  return r;
}

}  // namespace

int main() {
  int bad = 0;
  auto check = [&](bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    bad += ok ? 0 : 1;
  };
  const Result c = Run("car");
  check(c.waited_for_car && c.go_time > 17.0 && c.reach_time > 0 && !c.held_on_crosswalk,
        "car: waits for it, goes after it has passed, crosses without stopping");
  const Result n = Run("nocar");
  check(n.go_time > 0 && n.go_time < 14.0 && n.reach_time > 0 && !n.held_on_crosswalk,
        "no car: goes on arriving and crosses");
  const Result s = Run("stale");
  check(s.go_time < 0 && s.reach_time < 0 && s.arrive_time > 0, "no perception: waits at the curb, never goes");
  const Result t = Run("turned");
  check(t.aligned && t.go_time > 0 && t.reach_time > 0 && !t.held_on_crosswalk,
        "turned away: turns in place to face across, then goes");
  const Result j = Run("jump");
  check(j.unreleased && j.waited_for_car && j.go_time > 17.0 && j.cross_time > j.go_time && j.reach_time > 0,
        "pose jumps onto the crosswalk while waiting for the car: keeps waiting, goes after it has passed");
  const Result js = Run("jumpstale");
  check(js.unreleased && js.go_time < 0 && js.cross_time < 0 && js.reach_time < 0,
        "pose jumps onto the crosswalk with no perception: keeps waiting, never goes");
  const Result d = Run("deep");
  check(d.unreleased && d.cross_time >= 0 && d.cross_time < 0.1 && d.reach_time > 0,
        "starts 3 m onto the crosswalk with no perception: crosses (it is on the road)");
  std::printf("crosswalk_sim: %d failed\n", bad);
  return bad == 0 ? 0 : 1;
}
