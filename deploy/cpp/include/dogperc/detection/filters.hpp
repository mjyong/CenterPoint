// Post-detection sanity filters (detection/filters.py): too few current-sweep points in the box, box
// floating above / sunk below the local ground, a duplicate of a higher-scoring box of another class.
#pragma once

#include <string>
#include <vector>

#include "dogperc/config.hpp"
#include "dogperc/detections.hpp"

namespace dogperc {

// points of (M, 3) float32 xyz inside each box grown by `margin`
std::vector<int64_t> points_in_box_counts(const std::vector<float>& xyz, const std::vector<Box>& boxes, double margin);
// ground height under each box (NaN where the ring around it has too few occupied cells)
std::vector<double> local_ground_heights(const std::vector<float>& xyz, const std::vector<Box>& boxes, double cell,
                                         double ring, double percentile, int min_cells);
// keep mask: a box is dropped when its center lies in a higher-scoring box of another class
std::vector<char> cross_class_suppress(const Detections& d, double margin);
// what the filters test, per detection (filters.filter_measurements): box bottom above the local ground [m]
// and the current-sweep points in the box grown by box_margin; saved with the raw detections so the
// thresholds can be swept offline (pipeline_eval.py --detections raw)
void filter_measurements(const Detections& d, const float* points_det, size_t n, int stride,
                         const DetectionFilterConfig& cfg, std::vector<double>& bottom, std::vector<int64_t>& counts);
// why each detection is dropped ("" kept): floating, sunk, few points, duplicate
std::vector<std::string> filter_reasons(const Detections& d, const float* points_det, size_t n, int stride,
                                        const DetectionFilterConfig& cfg);
// points_det: (N, stride) det-frame cloud whose last column is the sweep lag (0 = current sweep)
Detections filter_detections(const Detections& d, const float* points_det, size_t n, int stride,
                             const DetectionFilterConfig& cfg);

}  // namespace dogperc
