// Detection-track assignment (tracking/association.py).
#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace lidar3d {

using Matches = std::vector<std::pair<int, int>>;   // (det, trk)

// repeatedly take the cheapest remaining pair; pairs costing more than max_cost (inf) are never matched
Matches greedy_match(const Eigen::MatrixXd& cost, double max_cost);
// optimal assignment (scipy.optimize.linear_sum_assignment on cost, invalid pairs 1e6), invalid pairs dropped
Matches hungarian_match(const Eigen::MatrixXd& cost, double max_cost);
// rectangular linear sum assignment, scipy's algorithm: (row, col) pairs sorted by row
Matches linear_sum_assignment(const Eigen::MatrixXd& cost);
Matches match(const Eigen::MatrixXd& cost, double max_cost, bool hungarian);

}  // namespace lidar3d
