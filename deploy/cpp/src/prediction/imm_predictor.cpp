#include "dogperc/prediction/imm_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dogperc {

Prediction Prediction::transform(const Mat4& T, const std::string& new_frame) const {
  Prediction o = *this;
  o.frame = new_frame;
  const Mat2 R = T.topLeftCorner<2, 2>();
  const Vec2 t = T.block<2, 1>(0, 3);
  for (size_t k = 0; k < modes.size(); ++k)
    for (size_t i = 0; i < modes[k].size(); ++i) {
      const Vec2& m = modes[k][i];
      o.modes[k][i] = Vec2(m[0] * R(0, 0) + m[1] * R(0, 1), m[0] * R(1, 0) + m[1] * R(1, 1)) + t;
      if (!covs.empty()) o.covs[k][i] = (R * covs[k][i]) * R.transpose();
    }
  return o;
}

void merge_modes(const std::vector<std::vector<Vec2>>& means, const std::vector<std::vector<Mat2>>& covs,
                 const std::vector<double>& probs, double merge_dist, std::vector<std::vector<Vec2>>& out_m,
                 std::vector<std::vector<Mat2>>& out_c, std::vector<double>& out_p) {
  const int M = static_cast<int>(probs.size());
  std::vector<int> order(M);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -probs[a] < -probs[b]; });
  std::vector<std::vector<int>> groups;
  for (int j : order) {
    bool placed = false;
    for (auto& g : groups) {
      double worst = 0.0;
      for (size_t t = 0; t < means[j].size(); ++t) {
        const Vec2 d = means[j][t] - means[g[0]][t];
        worst = std::max(worst, std::sqrt(d[0] * d[0] + d[1] * d[1]));
      }
      if (worst < merge_dist) {
        g.push_back(j);
        placed = true;
        break;
      }
    }
    if (!placed) groups.push_back({j});
  }
  out_m.clear();
  out_c.clear();
  out_p.clear();
  const size_t T = means.empty() ? 0 : means[0].size();
  for (const auto& g : groups) {
    double ps = 0.0;
    for (int j : g) ps += probs[j];
    std::vector<Vec2> m(T, Vec2::Zero());
    std::vector<Mat2> c(T, Mat2::Zero());
    for (int j : g)
      for (size_t t = 0; t < T; ++t) m[t] += (probs[j] / ps) * means[j][t];
    for (int j : g)
      for (size_t t = 0; t < T; ++t) {
        const Vec2 d = means[j][t] - m[t];
        c[t] += (probs[j] / ps) * (covs[j][t] + d * d.transpose());
      }
    out_m.push_back(m);
    out_c.push_back(c);
    out_p.push_back(ps);
  }
}

Prediction IMMPredictor::predict(const IMM& imm, int track_id, int label, double stamp) const {
  const Rollout r = imm.rollout(cfg_.horizon, cfg_.step);
  std::vector<std::vector<Vec2>> means;
  std::vector<std::vector<Mat2>> covs;
  for (size_t j = 0; j < r.means.size(); ++j) {
    std::vector<Vec2> m;
    std::vector<Mat2> c;
    for (size_t k = 0; k < r.means[j].size(); ++k) {
      m.push_back(r.means[j][k].head<2>());
      c.push_back(r.covs[j][k].topLeftCorner<2, 2>());
    }
    means.push_back(m);
    covs.push_back(c);
  }
  Prediction p;
  p.track_id = track_id;
  p.label = label;
  p.stamp = stamp;
  p.times = r.times;
  merge_modes(means, covs, r.mode_probs, cfg_.merge_dist, p.modes, p.covs, p.probs);
  return p;
}

std::vector<Prediction> IMMPredictor::operator()(const MultiObjectTracker& tracker) const {
  std::vector<Prediction> out;
  for (const Track* t : tracker.reported_tracks()) out.push_back(predict(t->imm, t->id, t->label, t->stamp));
  return out;
}

}  // namespace dogperc
