#ifndef D10_APPLICATION_POSE_REC_NODE_BODY_DETECTION_H_
#define D10_APPLICATION_POSE_REC_NODE_BODY_DETECTION_H_

#include <cstdint>
#include <string>

// Assume Header is defined in other header files
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace pose_rec_node_msg {
namespace msg {
struct BodyDetection {
    // Message header
    common_msg::msg::Header header;
    // Person ID ("unknown" when unknown)
    char id[64];
    // Probability of being male (0 - 1)
    float male_probability;
    // Person's name
    char name[1024];
    // Pose type (enumeration value)
    uint8_t pose_type;

    // Define enumeration constants
    static constexpr uint8_t POSE_CLASS_UNKNOWN = 0;  // Unknown pose
    static constexpr uint8_t POSE_CLASS_STANDING = 1;  // Standing
    static constexpr uint8_t POSE_CLASS_SITTING = 2;  // Sitting
    static constexpr uint8_t POSE_CLASS_LYING_DOWN = 3;  // Lying down
    static constexpr uint8_t POSE_CLASS_KNEELING = 4;  // Kneeling
    static constexpr uint8_t POSE_CLASS_CROUCHING = 5;  // Crouching
    static constexpr uint8_t POSE_CLASS_WALKING = 6;  // Walking
    static constexpr uint8_t POSE_CLASS_RUNNING = 7;  // Running
    static constexpr uint8_t POSE_CLASS_JUMPING = 8;  // Jumping
    static constexpr uint8_t POSE_CLASS_DANCING = 9;  // Dancing
    static constexpr uint8_t POSE_CLASS_RAISING_HANDS = 10;  // Raising hands
    static constexpr uint8_t POSE_CLASS_WAVING = 11;  // Waving
    static constexpr uint8_t POSE_CLASS_POINTING = 12;  // Pointing direction
    static constexpr uint8_t POSE_CLASS_CARRYING = 13;  // Carrying items
    static constexpr uint8_t POSE_CLASS_PUSHING = 14;  // Pushing
    static constexpr uint8_t POSE_CLASS_PULLING = 15;  // Pulling
    static constexpr uint8_t POSE_CLASS_BENDING = 16;  // Bending
    static constexpr uint8_t POSE_CLASS_CLIMBING = 17;  // Climbing
    static constexpr uint8_t POSE_CLASS_FALLING = 18;  // Falling
    static constexpr uint8_t POSE_CLASS_FIGHTING = 19;  // Fighting posture
    static constexpr uint8_t POSE_CLASS_GESTURING = 20;  // Gesturing
};

} // namespace msg
} // namespace pose_rec_node_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_POSE_REC_NODE_BODY_DETECTION_H_
