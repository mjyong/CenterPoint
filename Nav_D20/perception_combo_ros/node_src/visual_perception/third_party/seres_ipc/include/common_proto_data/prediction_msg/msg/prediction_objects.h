#ifndef D10_APPLICATION_PREDICTION_PREDICTIONOBJECTS_H_
#define D10_APPLICATION_PREDICTION_PREDICTIONOBJECTS_H_

#include <cstdint>
#include <string>
#include <vector>

// Assume Header, TrajectoryPrediction, DetectionObjectArray, and Point are defined in other header files
#include "common_msg/msg/header.h"
#include "trajectory_prediction.h"
#include "visual_perception_msg/msg/detection_object_array.h"
#include "common_msg/msg/point.h"

#pragma pack(1)

namespace seres_ipc {
namespace prediction_msg {
namespace msg {
struct PredictionObjects {
    // Message header
    common_msg::msg::Header header;
    // Trajectory prediction array
    TrajectoryPrediction trajectories[1024];
    // Detection object array
    visual_perception_msg::msg::DetectionObjectArray det_obj_arr;
    // Stopping point
    common_msg::msg::Point stopping_point;
    // Target lane ID
    char target_lane_id[1024];
    // Confidence
    double confidence;
};
} // namespace msg
} // namespace prediction_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_PREDICTION_PREDICTIONOBJECTS_H_
