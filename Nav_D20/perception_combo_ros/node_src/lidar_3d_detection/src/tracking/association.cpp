#include "lidar3d/tracking/association.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "lidar3d/common.hpp"

namespace lidar3d {

Matches greedy_match(const Eigen::MatrixXd& cost, double max_cost) {
  Matches out;
  if (cost.size() == 0) return out;
  std::vector<std::pair<int, int>> cand;   // np.nonzero order: row major
  for (int d = 0; d < cost.rows(); ++d)
    for (int t = 0; t < cost.cols(); ++t)
      if (cost(d, t) <= max_cost) cand.emplace_back(d, t);
  std::vector<int> order(cand.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return cost(cand[a].first, cand[a].second) < cost(cand[b].first, cand[b].second);
  });
  std::vector<char> used_d(cost.rows(), 0), used_t(cost.cols(), 0);
  for (int k : order) {
    const auto [d, t] = cand[k];
    if (used_d[d] || used_t[t]) continue;
    used_d[d] = used_t[t] = 1;
    out.emplace_back(d, t);
  }
  return out;
}

Matches linear_sum_assignment(const Eigen::MatrixXd& cost_in) {
  // shortest augmenting paths (Crouse 2016), as scipy/optimize/rectangular_lsap
  const bool transpose = cost_in.rows() > cost_in.cols();
  const Eigen::MatrixXd cost = transpose ? Eigen::MatrixXd(cost_in.transpose()) : cost_in;
  const int nr = static_cast<int>(cost.rows()), nc = static_cast<int>(cost.cols());
  Matches out;
  if (nr == 0 || nc == 0) return out;
  std::vector<double> u(nr, 0.0), v(nc, 0.0), spc(nc);
  std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
  std::vector<char> SR(nr), SC(nc);
  for (int cur = 0; cur < nr; ++cur) {
    double min_val = 0.0;
    int num_remaining = nc;
    for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
    std::fill(SR.begin(), SR.end(), 0);
    std::fill(SC.begin(), SC.end(), 0);
    std::fill(spc.begin(), spc.end(), kInf);
    int sink = -1, i = cur;
    while (sink == -1) {
      int index = -1;
      double lowest = kInf;
      SR[i] = 1;
      for (int it = 0; it < num_remaining; ++it) {
        const int j = remaining[it];
        const double r = min_val + cost(i, j) - u[i] - v[j];
        if (r < spc[j]) {
          path[j] = i;
          spc[j] = r;
        }
        if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) {
          lowest = spc[j];
          index = it;
        }
      }
      min_val = lowest;
      if (min_val == kInf) throw Error("linear_sum_assignment: cost matrix is infeasible");
      const int j = remaining[index];
      if (row4col[j] == -1) sink = j;
      else i = row4col[j];
      SC[j] = 1;
      remaining[index] = remaining[--num_remaining];
    }
    u[cur] += min_val;
    for (int r = 0; r < nr; ++r)
      if (SR[r] && r != cur) u[r] += min_val - spc[col4row[r]];
    for (int j = 0; j < nc; ++j)
      if (SC[j]) v[j] -= min_val - spc[j];
    int j = sink;
    while (true) {
      const int r = path[j];
      row4col[j] = r;
      std::swap(col4row[r], j);
      if (r == cur) break;
    }
  }
  if (!transpose) {
    for (int r = 0; r < nr; ++r) out.emplace_back(r, col4row[r]);
  } else {
    // rows of the original are the columns here: sort by them
    std::vector<int> idx(nr);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return col4row[a] < col4row[b]; });
    for (int r : idx) out.emplace_back(col4row[r], r);
  }
  return out;
}

Matches hungarian_match(const Eigen::MatrixXd& cost, double max_cost) {
  Matches out;
  if (cost.size() == 0) return out;
  Eigen::MatrixXd c = cost;
  for (int i = 0; i < c.rows(); ++i)
    for (int j = 0; j < c.cols(); ++j)
      if (!(cost(i, j) <= max_cost)) c(i, j) = 1e6;
  for (const auto& [r, k] : linear_sum_assignment(c))
    if (cost(r, k) <= max_cost) out.emplace_back(r, k);
  return out;
}

Matches match(const Eigen::MatrixXd& cost, double max_cost, bool hungarian) {
  return hungarian ? hungarian_match(cost, max_cost) : greedy_match(cost, max_cost);
}

}  // namespace lidar3d
