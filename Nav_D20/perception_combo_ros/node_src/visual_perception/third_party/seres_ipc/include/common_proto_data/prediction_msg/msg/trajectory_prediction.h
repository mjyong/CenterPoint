#ifndef D10_APPLICATION_PREDICTION_TRAJECTORY_PREDICTION_H_
#define D10_APPLICATION_PREDICTION_TRAJECTORY_PREDICTION_H_

#include <cstdint>
#include "prediction_msg/msg/trajectory_point.h"

#pragma pack(1)

namespace seres_ipc {
namespace prediction_msg {
namespace msg {
struct TrajectoryPrediction {
    // Array of trajectory points
    TrajectoryPoint points[20480];
    // Probability
    float probability;
    // Duration
    float duration;
    // Trajectory type
    uint8_t trajectory_type;

    // Define enumeration constants
    static constexpr uint8_t PREDICTION_UNKNOWN = 0;
    static constexpr uint8_t PREDICTION_LANE_KEEPING = 1; // Lane keeping
    static constexpr uint8_t PREDICTION_LANE_CHANGE_LEFT = 2; // Lane change left
    static constexpr uint8_t PREDICTION_LANE_CHANGE_RIGHT = 3; // Lane change right
    static constexpr uint8_t PREDICTION_TURN_LEFT = 4; // Turn left
    static constexpr uint8_t PREDICTION_TURN_RIGHT = 5; // Turn right
    static constexpr uint8_t PREDICTION_U_TURN = 6; // U-turn
    static constexpr uint8_t PREDICTION_ACCELERATING = 9; // Accelerating
    static constexpr uint8_t PREDICTION_DECELERATING = 10; // Decelerating
};
} // namespace msg
} // namespace prediction_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_PREDICTION_TRAJECTORY_PREDICTION_H_
